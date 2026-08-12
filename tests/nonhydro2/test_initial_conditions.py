"""Initial conditions (nh.initial_conditions).

Validates the prescribed-spectra random states on the nonhydro
tiers: family purity through the projections (periodic, walled
vertical and the walled-y channel), bitwise seed determinism, the
normalization convention, and the taught errors (Kelvin without
walls, the nonphysical constraint selection).

The named analytic ports: single_wave and kelvin_wave phase-rotate
exactly in the linear model, the wave package stays localized and
wave-pure, the geostrophically projected jets are steady, and the
coherent eddy is divergence-free and exactly balanced (its projected
tendency is machine zero on every topology, so it needs no vortical
projection). Both eddy branches serve every topology: the vorticity
route reproduces its prescribed Gaussian on all six (exactly where a
wall fixes the gauge, minus the domain mean where the horizontal is
fully periodic).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0
DT = 1e-3
COMPONENTS = ("u", "v", "w", "b")


def make_model(*, periodic_x=True, periodic_y=True, periodic_z=True,
               family=None, n=N, y_extent=(0.0, 1.0), advection=False,
               buoyancy=True, beta=None, nondimensional=False):
    """Build a small linear nonhydro model (walls as requested)."""
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 2 * np.pi),
                                     periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, y_extent,
                                     periodic=periodic_y, name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 2 * np.pi),
                                     periodic=periodic_z, name="z")
    # device_ids=(0,) keeps every axis local on any device count: the
    # prescribed-spectra ICs and eigenmode projections synthesize through
    # the naive (GSPMD) transform, which is a Tier-1 taught error on a
    # sharded transform axis. A follow-up phase re-legalizes the 3-D
    # channel synthesis on several devices; until then the math is tested
    # single-device (unchanged on the default suite).
    grid = fr.spatial.Grid((mx, my, mz), device_ids=(0,))
    # family=None follows the grid default: since the 2026-07-16 ruling
    # a periodic OR walled grid is finite-volume by default (only mapped
    # / immersed stay nodal), so the family=None b of the stratification
    # follows the model uniformly (no mixed nodal-b-on-FV-velocities
    # corner). Since stage F5 the analytic walled-vertical eigenmode kit
    # runs on both families, so the walled-vertical fixture covers both.
    if nondimensional:
        return nh.Model(
            grid=grid,
            core=nh.Core(aspect_ratio=(DSQR) ** 0.5, family=family),
            time_stepper=AdamBashforth(DT, order=3),
            scaling=fr.scaling.Advective(),
            coriolis=nh.FPlaneCoriolis(rossby_number=1.0 / F0),
            buoyancy=nh.ConstantStratification(
                froude_number=1.0 / N2 ** 0.5),
            advection=advection)
    coriolis = (nh.FPlaneCoriolis(f0=F0) if beta is None
                else nh.BetaPlaneCoriolis(f0=F0, beta=beta))
    return nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=(DSQR) ** 0.5, family=family),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=coriolis,
        buoyancy=(nh.ConstantStratification(n2=N2) if buoyancy
                  else None),
        advection=advection or not buoyancy)


@pytest.fixture(scope="module")
def periodic():
    """One fully periodic model + analytic eigenmodes (shared)."""
    model = make_model()
    return model, nh.eigenmodes.from_model(model)


@pytest.fixture(scope="module", params=["nodal", "fv"])
def walled(request):
    """One rigid-lid (walled z) analytic eigenmode set (shared).

    Since stage F5 the analytic walled-vertical eigenmode kit runs on
    both C-grid families (the FV kit mints its own BC-tagged CellAvg
    analysis spaces; test_fv_default::test_walled_fv_eigenmodes_build).
    """
    return nh.eigenmodes.from_model(
        make_model(periodic_z=False, family=request.param))


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
    omega, z0 = nh.single_wave(em, k, "wave+", phase=0.3)
    model.set_state(z0)
    steps = 20
    model.advance(steps)
    _, zt = nh.single_wave(em, k, "wave+",
                           phase=0.3 + omega * steps * DT)
    assert _absmax(zt, z0) > 0.01
    assert _absmax(model.state, zt) < 1e-4


def test_single_wave_is_the_mode_accessor(periodic, walled):
    model, em = periodic
    omega, z = nh.single_wave(model, {"x": 2, "y": 1, "z": 1},
                              "wave-", phase=0.7)
    omega_em, z_em = em.mode("wave-", {"x": 2, "y": 1, "z": 1},
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
        em, {"x": 2, "y": 0, "z": 1}, "wave+",
        envelope=nh.gaussian(pos={"x": np.pi},
                                      width={"x": 1.5}))
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


def test_wave_package_gaussian_helper_matches_a_plain_callable(
        periodic):
    _, em = periodic
    helper = nh.gaussian(
        pos={"x": np.pi, "z": np.pi}, width={"x": 1.5, "z": 1.5})

    def plain(x, z):
        return (jnp.exp(-((x - np.pi) ** 2) / 1.5 ** 2)
                * jnp.exp(-((z - np.pi) ** 2) / 1.5 ** 2))

    _, za = nh.wave_package(em, {"x": 2, "y": 0, "z": 1},
                            envelope=helper)
    _, zb = nh.wave_package(em, {"x": 2, "y": 0, "z": 1},
                            envelope=plain)
    assert _same(za, zb)


def test_wave_package_taught_errors(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="same"):
        nh.gaussian(pos={"x": 1.0}, width={"y": 1.0})
    with pytest.raises(ValueError, match="does not have"):
        nh.wave_package(
            em, {"x": 2, "y": 0, "z": 1},
            envelope=nh.gaussian(pos={"q": 1.0},
                                          width={"q": 1.0}))
    with pytest.raises(ValueError, match="names no coordinate"):
        nh.wave_package(em, {"x": 2, "y": 0, "z": 1},
                        envelope=lambda: 1.0)


# ================================================================
#  wave_package traveling= (single-sided packets on walled z)
# ================================================================
_DRIFT_K = {"x": 4, "y": 0, "z": 5}
_DRIFT_ENVELOPE = {"pos": {"x": 500.0, "z": 500.0},
                   "width": {"x": 220.0, "z": 220.0}}


@pytest.fixture(scope="module")
def drift_model():
    """Build a walled-z slice sized to show drift (48 x 1 x 32)."""
    mx = fr.spatial.meshes.IntervalMesh(48, (0.0, 2000.0),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(1, (0.0, 1.0),
                                     periodic=True, name="y")
    mz = fr.spatial.meshes.IntervalMesh(32, (0.0, 1000.0),
                                     periodic=False, name="z")
    grid = fr.spatial.Grid((mx, my, mz), device_ids=(0,))
    return nh.Model(
        grid=grid,
        coriolis=nh.FPlaneCoriolis(f0=1e-4),
        buoyancy=nh.ConstantStratification(n2=2.5e-5),
        advection=False,
        time_stepper=AdamBashforth(60.0, order=3))


def _b_centroid(model):
    b = np.asarray(model.state.b.data)[:, 0, :]
    profile = (b ** 2).sum(axis=0)
    centres = (np.arange(b.shape[1]) + 0.5) * (1000.0 / b.shape[1])
    return float((profile * centres).sum() / profile.sum())


@pytest.mark.parametrize("direction", [
    pytest.param(-1, id="down"), pytest.param(1, id="up")])
def test_wave_package_traveling_drifts_the_requested_way(
        drift_model, direction):
    omega, packet = nh.wave_package(
        drift_model, _DRIFT_K, "wave+",
        envelope=nh.gaussian(**_DRIFT_ENVELOPE),
        traveling={"z": direction})
    assert omega > 0.0
    drift_model.reset()
    drift_model.set_state(packet)
    start = _b_centroid(drift_model)
    drift_model.run(runlen=2400.0, progress=False)
    moved = _b_centroid(drift_model) - start
    # the continuum group drift is cg_z * t = 281 m (measured
    # 271-277 m at this resolution); the sign is the request
    assert np.sign(moved) == direction
    assert 0.7 * 281.0 < abs(moved) < 1.1 * 281.0


def test_wave_package_traveling_is_wave_pure(drift_model):
    em = nh.eigenbasis(drift_model)
    _, packet = nh.wave_package(
        em, _DRIFT_K, "wave+",
        envelope=nh.gaussian(**_DRIFT_ENVELOPE),
        traveling={"z": -1})
    total = _energy(packet)
    wave = _energy(nh.transforms.WaveProjection(em)(packet))
    vort = _energy(nh.transforms.VorticalProjection(em)(packet))
    assert wave / total > 1.0 - 1e-9
    assert vort / total < 1e-9


def test_wave_package_traveling_resolves_a_one_sided_slope(
        drift_model):
    # at the top of the wave lattice (m = nz - 1) the m + 1 neighbor
    # is the structurally absent buoyancy-top stratum, so the group
    # slope falls back to the one-sided difference
    omega, packet = nh.wave_package(
        drift_model, {"x": 4, "y": 0, "z": 31}, "wave+",
        envelope=nh.gaussian(**_DRIFT_ENVELOPE),
        traveling={"z": -1})
    assert omega > 0.0
    assert all(np.isfinite(np.asarray(packet[c].data)).all()
               for c in COMPONENTS)


def test_wave_package_traveling_needs_a_represented_neighbor():
    # nz = 2 leaves one wave stratum (m = 1) with both neighbors
    # structurally absent (m = 0 barotropic, m = 2 buoyancy top)
    mx = fr.spatial.meshes.IntervalMesh(8, (0.0, 2000.0),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(1, (0.0, 1.0),
                                     periodic=True, name="y")
    mz = fr.spatial.meshes.IntervalMesh(2, (0.0, 1000.0),
                                     periodic=False, name="z")
    grid = fr.spatial.Grid((mx, my, mz), device_ids=(0,))
    model = nh.Model(
        grid=grid,
        coriolis=nh.FPlaneCoriolis(f0=1e-4),
        buoyancy=nh.ConstantStratification(n2=2.5e-5),
        advection=False,
        time_stepper=AdamBashforth(60.0, order=3))
    with pytest.raises(ValueError, match="cannot resolve"):
        nh.wave_package(
            model, {"x": 2, "y": 0, "z": 1},
            envelope=nh.gaussian(pos={"z": 500.0},
                                          width={"z": 300.0}),
            traveling={"z": -1})


def test_wave_package_traveling_taught_errors(periodic, drift_model):
    _, em = periodic
    envelope = nh.gaussian(**_DRIFT_ENVELOPE)
    with pytest.raises(ValueError, match="bounded axes only"):
        nh.wave_package(
            em, {"x": 2, "y": 0, "z": 1},
            envelope=nh.gaussian(
                pos={"x": np.pi, "z": np.pi},
                width={"x": 1.0, "z": 1.0}),
            traveling={"z": -1})
    with pytest.raises(ValueError, match="does not vary"):
        nh.wave_package(
            drift_model, _DRIFT_K,
            envelope=nh.gaussian(pos={"x": 500.0},
                                          width={"x": 220.0}),
            traveling={"z": -1})
    with pytest.raises(ValueError, match="drift signs"):
        nh.wave_package(drift_model, _DRIFT_K, envelope=envelope,
                        traveling={"z": 0})
    with pytest.raises(ValueError, match="does not propagate"):
        nh.wave_package(drift_model, _DRIFT_K, "vortical",
                        envelope=envelope, traveling={"z": -1})
    with pytest.raises(ValueError, match="no oscillation"):
        nh.wave_package(drift_model, {"x": 4, "y": 0, "z": 0},
                        envelope=envelope, traveling={"z": -1})


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
#  coherent_eddy (the geostrophic Gaussian eddy)
# ================================================================
def _structure(zz):
    """Return a resolved, vertically periodic structure."""
    return 1.0 + 0.5 * jnp.cos(zz)


def _balance(model, state):
    """Projected tendency, relative to the raw Coriolis tendency."""
    tendency = model.tendency(state, constraints=True)
    raw = model.tendency(state, constraints=False)
    scale = max(float(np.abs(np.asarray(raw[c].data)).max())
                for c in ("u", "v"))
    residual = max(
        float(np.abs(np.asarray(tendency[c].data)).max())
        for c in state.components)
    return residual / scale


@pytest.fixture(scope="module")
def eddy_models():
    """Walled and nodal eddy models (shared; periodic is a fixture)."""
    return {
        "walled z": make_model(periodic_z=False),
        "walled x": make_model(periodic_x=False),
        "nodal": make_model(family="nodal"),
    }


@pytest.fixture(scope="module")
def square_model():
    """Return a square-domain model (the eddy is round on it)."""
    return make_model(n=16, y_extent=(0.0, 2 * np.pi))


#: The six topologies the vorticity inversion must serve.
_TOPOLOGIES = {
    "periodic": (True, True, True),
    "channel-x": (False, True, True),
    "box-xy": (False, False, True),
    "periodic+lid": (True, True, False),
    "channel-x+lid": (False, True, False),
    "box-xy+lid": (False, False, False),
}


@pytest.fixture(scope="module")
def topology_models():
    """One square-domain model per horizontal/vertical topology."""
    return {
        label: make_model(n=16, y_extent=(0.0, 2 * np.pi),
                          periodic_x=px, periodic_y=py, periodic_z=pz)
        for label, (px, py, pz) in _TOPOLOGIES.items()}


def _eddy_corner(model):
    """Return the eddy's vorticity corner (one DOF in the vertical)."""
    state = model.state
    mesh_z = next(m for m in model.grid.factors if "z" in m.names)
    return state["u"].function_space.bare.replace(
        y=state["v"].function_space.bare.factor("y"), z=mesh_z.constant)


def _prescribed_gaussian(model, width):
    """Sample coherent_eddy's own bump on the vorticity corner."""
    grid = model.grid
    x0, x1 = (float(v) for v in grid.factors[0].extent)
    y0, y1 = (float(v) for v in grid.factors[1].extent)
    lx = x1 - x0

    def init(x, y):
        return jnp.exp(
            -((x - x0 - 0.5 * lx) ** 2
              + (y - y0 - 0.5 * (y1 - y0)) ** 2) / (width * lx) ** 2)

    return grid.create_field(_eddy_corner(model), init=init, name="zeta")


@pytest.mark.parametrize("gauss_field",
                         ["vorticity", "streamfunction"])
def test_eddy_is_divergence_free(periodic, gauss_field):
    model, _ = periodic
    z = nh.coherent_eddy(model, width=0.15, gauss_field=gauss_field,
                         vertical_structure=_structure)
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    div = (z["u"].diff("x") + z["v"].diff("y")
           + z["w"].diff("z")).data
    assert umax > 0.0
    assert float(np.abs(np.asarray(div)).max()) < 1e-12 * umax
    assert float(np.abs(np.asarray(z["w"].data)).max()) == 0.0


def test_eddy_barotropic_has_no_buoyancy(periodic):
    model, _ = periodic
    z = nh.coherent_eddy(model, width=0.15)
    assert float(np.abs(np.asarray(z["b"].data)).max()) == 0.0
    assert max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v")) > 0.0


def test_eddy_periodic_is_exactly_balanced(periodic):
    """The projected tendency is machine zero: an exact steady state."""
    model, _ = periodic
    for structure in (None, _structure):
        z = nh.coherent_eddy(model, width=0.15,
                             vertical_structure=structure)
        model.set_state(z)
        assert _balance(model, z) < 1e-12


@pytest.mark.parametrize("gauss_field",
                         ["vorticity", "streamfunction"])
@pytest.mark.parametrize("topology",
                         ["walled z", "walled x", "nodal"])
def test_eddy_is_exactly_balanced_on_every_topology(eddy_models,
                                                    topology,
                                                    gauss_field):
    model = eddy_models[topology]
    z = nh.coherent_eddy(model, width=0.15, gauss_field=gauss_field,
                         vertical_structure=_structure)
    model.set_state(z)
    assert _balance(model, z) < 1e-12


def test_eddy_stays_steady_under_the_linear_model(periodic):
    model, _ = periodic
    z0 = nh.coherent_eddy(model, width=0.15,
                          vertical_structure=_structure)
    model.set_state(z0)
    model.run(steps=10, progress=False)
    scale = max(float(np.abs(np.asarray(z0[c].data)).max())
                for c in ("u", "v", "b"))
    drift = max(
        float(np.abs(np.asarray((model.state[c] - z0[c]).data)).max())
        for c in ("u", "v", "b"))
    assert drift < 1e-12 * scale
    assert float(np.abs(np.asarray(model.state["w"].data)).max()) < 1e-14


@pytest.mark.parametrize("gauss_field",
                         ["vorticity", "streamfunction"])
def test_eddy_needs_no_vortical_projection(periodic, gauss_field):
    """The construction is already exactly vortical (no radiation)."""
    model, em = periodic
    z = nh.coherent_eddy(model, width=0.15, gauss_field=gauss_field,
                         vertical_structure=_structure)
    assert _energy(nh.transforms.WaveProjection(em)(z)) < 1e-24 * _energy(z)
    # ... so the vortical projection is a no-op, not a correction
    kept = nh.transforms.VorticalProjection(em)(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in COMPONENTS)
    assert max(float(np.abs(np.asarray((kept[c] - z[c]).data)).max())
               for c in COMPONENTS) < 1e-12 * scale


def test_eddy_signs(square_model):
    """Check psi is the standard geostrophic streamfunction."""
    high = nh.coherent_eddy(square_model, width=0.1,
                            gauss_field="streamfunction")
    # a positive streamfunction is a pressure high, i.e. an anticyclone
    zeta = np.asarray(high.rel_vort_z.data)
    assert zeta.min() < 0.0
    assert abs(zeta.min()) > 4.0 * zeta.max()
    flipped = nh.coherent_eddy(square_model, width=0.1, amplitude=-1.0)
    assert np.allclose(np.asarray(flipped["u"].data),
                       -np.asarray(high["u"].data))


@pytest.mark.parametrize("label", list(_TOPOLOGIES))
def test_eddy_vorticity_serves_every_topology(topology_models, label):
    """Prescribed zeta comes back, exactly where a wall fixes the gauge."""
    periodic_x, periodic_y, _ = _TOPOLOGIES[label]
    model = topology_models[label]
    width = 0.12
    z = nh.coherent_eddy(model, width=width, gauss_field="vorticity")
    got = np.asarray(z.rel_vort_z.data).copy()
    want = np.broadcast_to(
        np.asarray(_prescribed_gaussian(model, width).data),
        got.shape).copy()
    scale = np.abs(want).max()
    if periodic_x and periodic_y:
        # the periodic gauge: a periodic domain admits no net
        # vorticity, so the bump comes back minus its own area
        # fraction pi * width^2 (the domain is square here)
        assert (np.abs(want - got).max() / scale
                == pytest.approx(np.pi * width ** 2, rel=1e-3))
        want -= want.mean(axis=(0, 1), keepdims=True)
        got -= got.mean(axis=(0, 1), keepdims=True)
    assert np.abs(want - got).max() / scale < 1e-11
    # divergence-free, with no wall-normal degree of freedom at all
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    divergence = (z["u"].diff("x") + z["v"].diff("y")
                  + z["w"].diff("z")).data
    assert float(np.abs(np.asarray(divergence)).max()) < 1e-12 * umax
    assert float(np.abs(np.asarray(z["w"].data)).max()) == 0.0
    if not periodic_x:
        cells = model.grid.factors[0].n_cells
        assert z["u"].function_space.shape[0] == cells - 1
        # the wall-adjacent columns close on u_wall = 0: nothing
        # leaks through the wall the space has no DOF on
        horizontal = np.asarray(
            (z["u"].diff("x") + z["v"].diff("y")).data)
        assert np.abs(horizontal[0]).max() < 1e-12 * umax
        assert np.abs(horizontal[-1]).max() < 1e-12 * umax


def test_eddy_vorticity_inverts_a_two_dimensional_operand(
        topology_models, monkeypatch):
    """F multiplies psi after the inversion, so the vertical is 1 DOF."""
    seen = []
    inversion = nh.initial_conditions.invert_negative_laplacian

    def spy(field, *, axes):
        seen.append(field.function_space.shape)
        return inversion(field, axes=axes)

    monkeypatch.setattr(nh.initial_conditions,
                        "invert_negative_laplacian", spy)
    model = topology_models["box-xy+lid"]
    z = nh.coherent_eddy(model, width=0.12, gauss_field="vorticity",
                         vertical_structure=_structure)
    assert len(seen) == 1
    # a 3-D operand would pay a DST-II pair along a vertical the
    # symbol never reads (19x the cost at 128^2 x 32)
    assert seen[0][2] == 1
    assert z["u"].function_space.shape[2] > 1


def test_eddy_vorticity_baroclinic_is_the_barotropic_one_times_f(
        topology_models):
    """The structure scales the recovered vorticity level by level."""
    model = topology_models["channel-x+lid"]
    flat = np.asarray(nh.coherent_eddy(
        model, width=0.12, gauss_field="vorticity").rel_vort_z.data)
    tall = np.asarray(nh.coherent_eddy(
        model, width=0.12, gauss_field="vorticity",
        vertical_structure=_structure).rel_vort_z.data)
    i, j = np.unravel_index(np.abs(flat[:, :, 0]).argmax(),
                            flat.shape[:2])
    ratios = tall[i, j] / flat[i, j]        # the vertical profile
    for k, ratio in enumerate(ratios):
        assert (np.abs(tall[:, :, k] - ratio * flat[:, :, k]).max()
                < 1e-12 * np.abs(flat[:, :, k]).max())
    assert ratios.max() / ratios.min() > 1.5    # F really varies


def test_eddy_vorticity_branch_reproduces_its_gaussian(square_model):
    """The inversion is the exact inverse of the discrete curl."""
    z = nh.coherent_eddy(square_model, width=0.1,
                         gauss_field="vorticity")
    zeta = np.asarray(z.rel_vort_z.data)[:, :, 0]
    # the zero-mean gauge removes the domain average of the Gaussian
    mean = zeta.mean()
    assert abs(mean) < 1e-12
    nodes = square_model.grid.evaluation_nodes(
        z.rel_vort_z.function_space, "x").data
    xs = np.asarray(nodes).ravel()
    peak = np.abs(xs - np.pi).argmin()
    expected = 1.0 - (np.pi * (0.1 * 2 * np.pi) ** 2
                      / (2 * np.pi) ** 2)
    assert zeta[peak, peak] == pytest.approx(expected, rel=0.05)


def test_eddy_vertical_structure_is_surface_trapped(eddy_models):
    """A decaying structure gives a surface-trapped, warm-core eddy."""
    model = eddy_models["walled z"]
    top = float(np.asarray(model.grid.factors[2].extent[1]))
    z = nh.coherent_eddy(
        model, width=0.15,
        vertical_structure=lambda zz: jnp.exp((zz - top) / (0.3 * top)))
    speed = np.abs(np.asarray(z["u"].data)).max(axis=(0, 1))
    assert np.all(np.diff(speed) > 0.0)          # grows towards the top
    assert speed[-1] > 10.0 * speed[0]
    buoyancy = np.asarray(z["b"].data)
    # b = f0 d_z psi > 0 under a positive, upward-growing psi (warm core)
    assert buoyancy.min() > 0.0
    assert buoyancy.max(axis=(0, 1))[-1] > 10.0 * buoyancy.max(
        axis=(0, 1))[0]
    # a negative amplitude flips the core cold
    cold = nh.coherent_eddy(
        model, width=0.15, amplitude=-1.0,
        vertical_structure=lambda zz: jnp.exp((zz - top) / (0.3 * top)))
    assert np.asarray(cold["b"].data).max() < 0.0


def test_eddy_thermal_wind_is_second_order():
    """Check f d_z u = -d_y b closes at truncation order."""
    errors = []
    for size in (8, 16):
        model = make_model(n=size, y_extent=(0.0, 2 * np.pi))
        z = nh.coherent_eddy(model, width=0.15,
                             vertical_structure=_structure)
        lhs = (F0 * z["u"].diff("z")).to(z["v"].function_space.bare)
        rhs = -z["b"].diff("y")
        errors.append(
            float(np.abs(np.asarray((lhs - rhs).data)).max())
            / float(np.abs(np.asarray(rhs.data)).max()))
    assert errors[0] > 1e-3
    assert errors[1] < 0.35 * errors[0]


def test_eddy_reads_the_nondimensional_rotation():
    """Check f is eps/Ro on a nondimensional assembly."""
    model = make_model(nondimensional=True)
    z = nh.coherent_eddy(model, width=0.15,
                         vertical_structure=_structure)
    model.set_state(z)
    assert float(np.abs(np.asarray(z["b"].data)).max()) > 0.0
    assert _balance(model, z) < 1e-12


@pytest.mark.parametrize("periodic_x", [True, False],
                         ids=["periodic", "walled-x"])
def test_eddy_survives_a_nonlinear_run(periodic_x):
    model = make_model(advection=True, periodic_x=periodic_x)
    z = nh.coherent_eddy(model, width=0.15, amplitude=0.1,
                         vertical_structure=_structure)
    model.set_state(z)
    model.run(steps=10, progress=False)
    for name in COMPONENTS:
        assert np.isfinite(np.asarray(model.state[name].data)).all()
    assert (float(np.abs(np.asarray(model.state["u"].data)).max())
            < 2.0 * float(np.abs(np.asarray(z["u"].data)).max()))


def test_eddy_taught_errors(periodic, eddy_models):
    model, em = periodic
    with pytest.raises(ValueError, match="unknown gauss_field"):
        nh.coherent_eddy(model, gauss_field="pressure")
    with pytest.raises(ValueError, match="not an eigenmodes"):
        nh.coherent_eddy(em)
    # the topology is not a reason: the vorticity branch inverts on a
    # walled horizontal too (test_eddy_vorticity_serves_every_topology)
    walled = nh.coherent_eddy(eddy_models["walled x"],
                              gauss_field="vorticity")
    assert float(np.abs(np.asarray(walled["v"].data)).max()) > 0.0
    with pytest.raises(ValueError, match="declares no 'b'"):
        nh.coherent_eddy(make_model(buoyancy=False),
                         vertical_structure=_structure)
    with pytest.raises(ValueError, match="constant Coriolis"):
        nh.coherent_eddy(make_model(beta=0.2),
                         vertical_structure=_structure)


# ================================================================
#  Distributed (sharded) analytic random-state (forced-4)
# ================================================================
def _periodic_model_at(device_ids, n=16):
    """Return a fully periodic nonhydro model at a device layout."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            n, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z"))
    return nh.Model(
        grid=fr.spatial.Grid(meshes, device_ids=device_ids),
        core=nh.Core(aspect_ratio=(DSQR) ** 0.5),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=nh.ConstantStratification(n2=N2),
        advection=False)


@pytest.mark.multi_device
def test_random_state_on_a_sharded_grid_is_device_invariant(
        forced_devices):
    # the analytic random-state synthesis builds its gains and Hermitian
    # random phases on the device-independent single-device coefficient
    # frame; the default nonhydro layout shards the half axis, so the
    # transpose engine re-designates the half axis and the synthesis routes
    # through the fused jax.shard_map backward via the Hermitian half-axis
    # re-expression (no gather) -- reproducing the one-device state to
    # floating point, real and sharded, on any device count.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 16
    many = nh.eigenmodes.from_model(_periodic_model_at(None, n))
    one = nh.eigenmodes.from_model(_periodic_model_at((0,), n))
    for family in ("vortical", "wave"):
        s_many = nh.random_state(many, family, seed=21)
        s_one = nh.random_state(one, family, seed=21)
        assert s_many["u"]._data.sharding.spec[0] == "devices"
        assert all(not np.iscomplexobj(np.asarray(s_many[c].data))
                   for c in COMPONENTS)
        err = max(
            float(np.abs(np.asarray(s_many[c].data)
                         - np.asarray(s_one[c].data)).max())
            for c in COMPONENTS)
        assert err < 1e-11


@pytest.mark.multi_device
def test_single_wave_on_a_sharded_grid_is_device_invariant(
        forced_devices):
    # em.mode (the single_wave accessor) shares the synthesis tail: the
    # Hermitian-closed single-mode column routes through the same fused
    # half-axis re-expression on a grid that shards the half axis, so the
    # sharded mode reproduces the one-device mode (frequency and field) to
    # floating point, real and sharded.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 16
    many = nh.eigenmodes.from_model(_periodic_model_at(None, n))
    one = nh.eigenmodes.from_model(_periodic_model_at((0,), n))
    k = {"x": 3, "y": 1, "z": 2}
    om_many, z_many = nh.single_wave(many, k, "wave+", phase=0.4)
    om_one, z_one = nh.single_wave(one, k, "wave+", phase=0.4)
    assert om_many == om_one
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    assert all(not np.iscomplexobj(np.asarray(z_many[c].data))
               for c in COMPONENTS)
    err = max(
        float(np.abs(np.asarray(z_many[c].data)
                     - np.asarray(z_one[c].data)).max())
        for c in COMPONENTS)
    assert err < 1e-11
