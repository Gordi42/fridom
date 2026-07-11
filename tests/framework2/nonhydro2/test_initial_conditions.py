"""Initial conditions (nh.initial_conditions).

Validates the prescribed-spectra random states on the nonhydro
tiers: family purity through the projections (periodic, walled
vertical and the walled-y channel), bitwise seed determinism, the
normalization convention, and the taught errors (Kelvin without
walls, the nonphysical constraint selection).

The named analytic ports: single_wave and kelvin_wave phase-rotate
exactly in the linear model, the wave package stays localized and
wave-pure, the geostrophically projected jets are steady, and the
coherent eddy is discretely divergence-free.
"""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0
DT = 1e-3
COMPONENTS = ("u", "v", "w", "b")


def make_model(*, periodic_y=True, periodic_z=True):
    """Build a small linear nonhydro model (walls as requested)."""
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    mz = fr.grid.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=periodic_z, name="z")
    return nh.Model(
        grid=fr.grid.Grid((mx, my, mz)), advection=False,
        dsqr=DSQR, coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(DT, order=3))


@pytest.fixture(scope="module")
def periodic():
    """One fully periodic model + analytic eigenmodes (shared)."""
    model = make_model()
    return model, nh.eigenmodes.from_model(model)


@pytest.fixture(scope="module")
def walled():
    """One rigid-lid (walled z) analytic eigenmode set (shared)."""
    return nh.eigenmodes.from_model(make_model(periodic_z=False))


@pytest.fixture(scope="module")
def channel_model():
    """One walled-y channel model (shared)."""
    return make_model(periodic_y=False)


@pytest.fixture(scope="module")
def channel(channel_model):
    """One walled-y channel eigenbasis (shared)."""
    return nh.eigenbasis(channel_model)


def _energy(state):
    weights = {"u": 1.0, "v": 1.0, "w": DSQR, "b": 1.0 / N2}
    return sum(
        float((np.asarray(state[c].data) ** 2).sum()) * weights[c]
        for c in COMPONENTS)


def _same(a, b):
    return all(
        np.array_equal(np.asarray(a[c].data), np.asarray(b[c].data))
        for c in COMPONENTS)


# ================================================================
#  Family purity through the projections
# ================================================================
def test_periodic_random_vortical_is_purely_vortical(periodic):
    model, em = periodic
    state = nh.random_vortical(em, seed=11)
    total = _energy(state)
    vort = _energy(nh.transforms.VorticalProjection(em)(state))
    wave = _energy(nh.transforms.WaveProjection(em)(state))
    assert vort / total > 1.0 - 1e-10
    assert wave / total < 1e-10
    # the factory accepts the model source as well
    assert _same(state, nh.random_vortical(model, seed=11))


def test_walled_random_waves_are_purely_wave(walled):
    state = nh.random_waves(walled, seed=11)
    total = _energy(state)
    wave = _energy(nh.transforms.WaveProjection(walled)(state))
    vort = _energy(nh.transforms.VorticalProjection(walled)(state))
    assert wave / total > 1.0 - 1e-10
    assert vort / total < 1e-10


def test_channel_random_states_stay_in_their_families(channel):
    eb = channel
    vortical = nh.random_vortical(eb, seed=3)
    assert (_energy(eb.projector("vortical")(vortical))
            / _energy(vortical) > 1.0 - 1e-10)
    waves = nh.random_waves(eb, seed=3)
    covered = (_energy(eb.projector("wave")(waves))
               + _energy(eb.projector("kelvin")(waves)))
    assert covered / _energy(waves) > 1.0 - 1e-10


# ================================================================
#  Determinism, normalization, realness
# ================================================================
def test_same_seed_is_bitwise_identical(walled):
    assert _same(nh.random_vortical(walled, seed=7),
                 nh.random_vortical(walled, seed=7))
    assert not _same(nh.random_vortical(walled, seed=7),
                     nh.random_vortical(walled, seed=8))


def test_normalization_and_realness(periodic):
    _, em = periodic
    state = nh.random_waves(em, seed=9)
    peak = max(float(np.abs(np.asarray(state[c].data)).max())
               for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(state[c].data))


def test_custom_spectrum_reweights_the_state(periodic):
    _, em = periodic
    default = nh.random_vortical(em, seed=2)
    steep = nh.random_vortical(
        em, seed=2,
        spectral_energy_density=lambda kx, ky, *_kz:
        nh.initial_conditions.geostrophic_energy_spectrum(
            kx, ky, d=9.0))
    assert not _same(default, steep)


# ================================================================
#  Taught errors
# ================================================================
def test_kelvin_needs_walls(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        nh.random_state(em, "kelvin", seed=1)


def test_unknown_family_is_taught(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="unknown mode family"):
        nh.random_state(em, "rossby", seed=1)


def test_constraint_selection_is_nonphysical(channel):
    with pytest.raises(ValueError, match="unknown or nonphysical"):
        nh.random_state(channel, "constraint", seed=1)


# ================================================================
#  single_wave and kelvin_wave (SingleWave / KelvinWave ports)
# ================================================================
def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


def test_single_wave_phase_rotates_in_the_linear_model(periodic):
    model, em = periodic
    k = {"x": 2, "y": 1, "z": 1}
    omega, z0 = nh.single_wave(em, k, s=1, phase=0.3)
    model.set_state(z0)
    steps = 20
    model.advance(steps)
    _, zt = nh.single_wave(em, k, s=1,
                           phase=0.3 + omega * steps * DT)
    assert _absmax(zt, z0) > 0.01
    assert _absmax(model.state, zt) < 1e-4


def test_single_wave_is_the_mode_accessor(periodic, walled):
    model, em = periodic
    omega, z = nh.single_wave(model, {"x": 2, "y": 1, "z": 1},
                              s=-1, phase=0.7)
    omega_em, z_em = em.mode(-1, {"x": 2, "y": 1, "z": 1},
                             phase=0.7)
    assert omega == omega_em
    assert _same(z, z_em)
    # the walled vertical resolves through the same analytic tier
    omega_w, z_w = nh.single_wave(walled, {"x": 2, "y": 1, "z": 2})
    assert omega_w > 0.0
    assert float(np.abs(np.asarray(z_w["u"].data)).max()) > 0.1


def test_single_wave_needs_the_analytic_tier(channel):
    with pytest.raises(ValueError, match="walled channel"):
        nh.single_wave(channel, {"x": 2, "y": 0, "z": 1})


def test_kelvin_wave_phase_rotates_in_the_channel(channel_model,
                                                  channel):
    omega, z0 = nh.kelvin_wave(channel, {"x": 2, "z": 1},
                               branch=1, phase=0.2)
    channel_model.set_state(z0)
    steps = 20
    channel_model.advance(steps)
    _, zt = nh.kelvin_wave(channel, {"x": 2, "z": 1}, branch=1,
                           phase=0.2 + omega * steps * DT)
    assert _absmax(zt, z0) > 0.01
    assert _absmax(channel_model.state, zt) < 1e-4


def test_kelvin_wave_defaults_the_bounded_ordinal(channel):
    omega, z = nh.kelvin_wave(channel, {"x": 2, "z": 1})
    omega_full, z_full = nh.kelvin_wave(
        channel, {"x": 2, "z": 1, "y": 0})
    assert omega == omega_full
    assert _same(z, z_full)


def test_kelvin_wave_needs_walls(periodic):
    model, _ = periodic
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        nh.kelvin_wave(model, {"x": 2, "z": 1})


# ================================================================
#  wave_package (the WavePackage port)
# ================================================================
def test_wave_package_localizes_and_stays_wave_pure(periodic):
    _, em = periodic
    omega, z = nh.wave_package(
        em, {"x": 2, "y": 0, "z": 1}, s=1,
        mask_pos={"x": np.pi}, mask_width={"x": 1.5})
    assert omega > 0.0
    # localized: the envelope suppresses the far side of the domain
    profile = np.abs(np.asarray(z["u"].data)).max(axis=(1, 2))
    assert profile.min() < 0.2 * profile.max()
    # polarized: the re-projection keeps the state wave-pure
    total = _energy(z)
    wave = _energy(nh.transforms.WaveProjection(em)(z))
    vort = _energy(nh.transforms.VorticalProjection(em)(z))
    assert wave / total > 1.0 - 1e-10
    assert vort / total < 1e-10


def test_wave_package_taught_errors(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="same"):
        nh.wave_package(em, {"x": 2, "y": 0, "z": 1},
                        mask_pos={"x": 1.0}, mask_width={"y": 1.0})
    with pytest.raises(ValueError, match="does not have"):
        nh.wave_package(em, {"x": 2, "y": 0, "z": 1},
                        mask_pos={"q": 1.0}, mask_width={"q": 1.0})


# ================================================================
#  Jets (BarotropicJet / Jet ports)
# ================================================================
def test_barotropic_jet_is_steady_when_projected(periodic):
    model, em = periodic
    z = nh.barotropic_jet(em)
    tendency = model.tendency(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in COMPONENTS)
    residual = max(
        float(np.abs(np.asarray(tendency[c].data)).max())
        for c in COMPONENTS)
    assert scale > 0.5
    assert residual < 1e-12 * scale


def test_barotropic_jet_raw_profile(periodic):
    _, em = periodic
    z = nh.barotropic_jet(em, wavenum=2, waveamp=0.1,
                          geo_proj=False)
    u = np.asarray(z["u"].data)
    ys = np.asarray(em.grid.evaluation_nodes(
        z["u"].function_space, "y").data).ravel()
    east = u[:, np.abs(ys - 0.75).argmin(), :]
    west = u[:, np.abs(ys - 0.25).argmin(), :]
    assert east.min() > 0.0
    assert west.max() < 0.0
    v = np.asarray(z["v"].data)
    xs = np.asarray(em.grid.evaluation_nodes(
        z["v"].function_space, "x").data).ravel()
    expected = 0.1 * np.abs(np.sin(2.0 * xs)).max()
    assert float(np.abs(v).max()) == pytest.approx(expected,
                                                   rel=1e-12)
    assert float(np.abs(np.asarray(z["w"].data)).max()) == 0.0


def test_jet_is_steady_when_projected(periodic):
    model, em = periodic
    z = nh.jet(em, pert_wavenum=2)
    tendency = model.tendency(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in COMPONENTS)
    residual = max(
        float(np.abs(np.asarray(tendency[c].data)).max())
        for c in COMPONENTS)
    assert scale > 0.1
    assert residual < 1e-12 * scale


def test_jet_carries_the_vertical_shear(periodic):
    _, em = periodic
    z = nh.jet(em, pert_strength=0.0, pert_wavenum=2,
               geo_proj=False, jet_width=0.16)
    u = np.asarray(z["u"].data)
    ys = np.asarray(em.grid.evaluation_nodes(
        z["u"].function_space, "y").data).ravel()
    zs = np.asarray(em.grid.evaluation_nodes(
        z["u"].function_space, "z").data).ravel()
    jy = np.abs(ys - 0.75).argmin()
    surface = u[0, jy, np.abs(zs).argmin()]
    middepth = u[0, jy, np.abs(zs - np.pi).argmin()]
    assert surface > 0.0
    assert middepth < 0.0
    assert surface == pytest.approx(-middepth, rel=1e-6)


def test_jets_need_the_analytic_tier(channel):
    with pytest.raises(ValueError, match="walled channel"):
        nh.barotropic_jet(channel)
    with pytest.raises(ValueError, match="walled channel"):
        nh.jet(channel)


# ================================================================
#  coherent_eddy (the CoherentEddy port)
# ================================================================
@pytest.mark.parametrize("gauss_field",
                         ["vorticity", "streamfunction"])
def test_eddy_is_divergence_free(periodic, gauss_field):
    _, em = periodic
    z = nh.coherent_eddy(em, width=0.15, gauss_field=gauss_field)
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    div = (z["u"].diff("x") + z["v"].diff("y")).data
    assert umax > 0.0
    assert float(np.abs(np.asarray(div)).max()) < 1e-12 * umax
    assert float(np.abs(np.asarray(z["w"].data)).max()) == 0.0
    assert float(np.abs(np.asarray(z["b"].data)).max()) == 0.0


def test_eddy_streamfunction_works_on_the_walled_vertical(walled):
    z = nh.coherent_eddy(walled, gauss_field="streamfunction")
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    div = (z["u"].diff("x") + z["v"].diff("y")).data
    assert umax > 0.0
    assert float(np.abs(np.asarray(div)).max()) < 1e-12 * umax


def test_eddy_taught_errors(periodic, walled, channel):
    _, em = periodic
    with pytest.raises(ValueError, match="unknown gauss_field"):
        nh.coherent_eddy(em, gauss_field="pressure")
    with pytest.raises(ValueError, match="fully periodic"):
        nh.coherent_eddy(walled, gauss_field="vorticity")
    with pytest.raises(ValueError, match="walled channel"):
        nh.coherent_eddy(channel)
