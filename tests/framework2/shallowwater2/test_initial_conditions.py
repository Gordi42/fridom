"""Initial conditions (sw.initial_conditions).

Validates the prescribed-spectra random states on both eigenmode
tiers: family purity (the projections recover ~100% of the energy),
the realized per-mode energy against the prescribed ``S/(pi k_h)``
convention (deterministic amplitudes: unit-modulus phases), bitwise
seed determinism, the normalization convention, the taught errors,
and the multi-device device-count invariance of the analytic path.

The named analytic ports: single_wave phase-rotates exactly in the
linear model, the geostrophically projected jet is steady, the
coherent eddy is discretely divergence-free and balanced, and the
equatorial wave satisfies the beta-plane eigen-relation to
discretization accuracy.
"""
from itertools import pairwise

import jax
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.model.eigenstates import (
    geostrophic_energy_spectrum,
)

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")
CSQR = 2.0


@pytest.fixture(scope="module")
def periodic():
    """One linear periodic model + analytic eigenmodes (shared)."""
    model = make_model(csqr=CSQR, f0=1.5, advection=False)
    return model, sw.eigenmodes.from_model(model)


@pytest.fixture(scope="module")
def channel():
    """One walled channel model + labeled eigenbasis (shared)."""
    model = make_model(make_grid(periodic_y=False), csqr=CSQR,
                       f0=1.5, advection=False)
    return model, sw.eigenbasis(model)


def _energy(state, csqr=CSQR):
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
    return sum(
        float((np.asarray(state[c].data) ** 2).sum()) * weights[c]
        for c in COMPONENTS)


def _same(a, b):
    return all(
        np.array_equal(np.asarray(a[c].data), np.asarray(b[c].data))
        for c in COMPONENTS)


# ================================================================
#  Family purity (the projections recover the state)
# ================================================================
def test_random_vortical_is_purely_vortical(periodic):
    _, em = periodic
    state = sw.random_vortical(em, seed=42)
    total = _energy(state)
    vort = _energy(sw.transforms.VorticalProjection(em)(state))
    wave = _energy(sw.transforms.WaveProjection(em)(state))
    assert vort / total > 1.0 - 1e-10
    assert wave / total < 1e-10


def test_random_waves_is_purely_wave(periodic):
    _, em = periodic
    state = sw.random_waves(em, seed=5)
    total = _energy(state)
    assert (_energy(sw.transforms.WaveProjection(em)(state)) / total
            > 1.0 - 1e-10)
    assert (_energy(sw.transforms.VorticalProjection(em)(state))
            / total < 1e-10)


def test_random_state_accepts_a_model_source(periodic):
    model, em = periodic
    from_model = sw.random_state(model, "vortical", seed=3)
    from_em = sw.random_state(em, "vortical", seed=3)
    assert _same(from_model, from_em)


# ================================================================
#  The realized spectrum follows the prescribed shape
# ================================================================
def test_realized_energy_follows_the_prescribed_spectrum(periodic):
    # unit-modulus phases make per-mode energies deterministic:
    # every populated mode carries S(k)/(pi k_h) (times a single
    # global normalization), so the per-mode ratio is constant and
    # the binned 1-D spectrum matches the prescribed shape exactly
    _, em = periodic
    state = sw.random_vortical(em, seed=42)
    coeff = {
        c: np.fft.fft(np.fft.rfft(np.asarray(state[c].data),
                                  axis=0), axis=1)
        for c in COMPONENTS}
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / CSQR}
    multiplicity = np.ones((N // 2 + 1, N))
    multiplicity[1:-1] *= 2.0
    energy = sum(multiplicity * weights[c] * np.abs(coeff[c]) ** 2
                 for c in COMPONENTS)
    kx = 2 * np.pi * np.fft.rfftfreq(N, 1.0 / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    kh = np.hypot(kx[:, None], ky[None, :])
    spectra = np.asarray(geostrophic_energy_spectrum(
        kx[:, None], ky[None, :]))
    expected = multiplicity * np.where(
        kh > 0, spectra / (np.pi * np.where(kh > 0, kh, 1.0)), 0.0)
    # only the far spectral tail is excluded against fp leakage
    mask = (kh > 0) & (expected > 1e-6 * expected.max())
    ratio = energy[mask] / expected[mask]
    assert ratio.max() / ratio.min() < 1.0 + 1e-6
    # the interpolation-Nyquist planes carry the steady
    # divergence-free stratum of the geostrophic family, so they
    # are populated with the same per-mode normalization as the
    # interior (they only sit in the spectral tail here)
    for pt in ((N // 2, 1), (1, N // 2), (N // 2, 0), (0, N // 2)):
        assert energy[pt] > 0.0
        assert abs(energy[pt] / expected[pt] / ratio.mean()
                   - 1.0) < 1e-6
    # binned 1-D spectrum: shape-correlated with S(k)
    bins = np.arange(0.5, kh.max() / (2 * np.pi)) * 2 * np.pi
    binned, prescribed = [], []
    for lo, hi in pairwise(bins):
        shell = (kh >= lo) & (kh < hi)
        if shell.any():
            binned.append(energy[shell].sum())
            prescribed.append(expected[shell].sum())
    corr = np.corrcoef(binned, prescribed)[0, 1]
    assert corr > 0.9999


# ================================================================
#  Determinism, normalization, realness
# ================================================================
def test_same_seed_is_bitwise_identical(periodic):
    _, em = periodic
    assert _same(sw.random_vortical(em, seed=7),
                 sw.random_vortical(em, seed=7))


def test_different_seeds_differ(periodic):
    _, em = periodic
    assert not _same(sw.random_vortical(em, seed=7),
                     sw.random_vortical(em, seed=8))


def test_custom_spectrum_reweights_the_state(periodic):
    _, em = periodic
    default = sw.random_vortical(em, seed=2)
    steep = sw.random_vortical(
        em, seed=2,
        spectral_energy_density=lambda kx, ky:
        geostrophic_energy_spectrum(kx, ky, d=9.0))
    assert not _same(default, steep)


def test_normalization_and_realness(periodic):
    _, em = periodic
    state = sw.random_waves(em, seed=9)
    peak = max(float(np.abs(np.asarray(state[c].data)).max())
               for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(state[c].data))


# ================================================================
#  Taught errors
# ================================================================
def test_kelvin_needs_walls(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        sw.random_state(em, "kelvin", seed=1)


def test_unknown_family_is_taught(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="unknown mode family"):
        sw.random_state(em, "rossby", seed=1)


# ================================================================
#  The channel tier
# ================================================================
def test_channel_random_vortical_is_purely_vortical(channel):
    _, eb = channel
    state = sw.random_vortical(eb, seed=11)
    total = _energy(state)
    vort = _energy(eb.projector("vortical")(state))
    assert vort / total > 1.0 - 1e-10
    assert _energy(eb.projector("wave")(state)) / total < 1e-10


def test_channel_random_waves_cover_wave_and_kelvin(channel):
    _, eb = channel
    state = sw.random_waves(eb, seed=11)
    total = _energy(state)
    wave = _energy(eb.projector("wave")(state))
    kelvin = _energy(eb.projector("kelvin")(state))
    assert (wave + kelvin) / total > 1.0 - 1e-10
    assert wave / total > 0.1
    assert kelvin / total > 0.1
    peak = max(float(np.abs(np.asarray(state[c].data)).max())
               for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)


def test_channel_random_state_is_deterministic(channel):
    _, eb = channel
    assert _same(sw.random_vortical(eb, seed=4),
                 sw.random_vortical(eb, seed=4))
    assert not _same(sw.random_vortical(eb, seed=4),
                     sw.random_vortical(eb, seed=5))


# ================================================================
#  single_wave (the SingleWave port)
# ================================================================
@pytest.fixture(scope="module")
def wave_setup():
    """One linear periodic model with a wave-resolving time step."""
    model = make_model(csqr=CSQR, f0=1.5, advection=False, dt=1e-3)
    return model, sw.eigenmodes.from_model(model)


def test_single_wave_phase_rotates_in_the_linear_model(wave_setup):
    model, em = wave_setup
    omega, z0 = sw.single_wave(em, {"x": 2, "y": 1}, s=1, phase=0.3)
    model.set_state(z0)
    steps = 20
    model.advance(steps)
    _, zt = sw.single_wave(em, {"x": 2, "y": 1}, s=1,
                           phase=0.3 + omega * steps * 1e-3)
    moved = max(
        float(np.abs(np.asarray(zt[c].data)
                     - np.asarray(z0[c].data)).max())
        for c in COMPONENTS)
    err = max(
        float(np.abs(np.asarray(model.state[c].data)
                     - np.asarray(zt[c].data)).max())
        for c in COMPONENTS)
    assert moved > 0.1
    assert err < 1e-3


def test_single_wave_is_the_mode_accessor(wave_setup):
    model, em = wave_setup
    omega, z = sw.single_wave(model, {"x": 3, "y": 2}, s=-1,
                              phase=0.7)
    omega_em, z_em = em.mode(-1, {"x": 3, "y": 2}, phase=0.7)
    assert omega == omega_em
    assert _same(z, z_em)


def test_single_wave_needs_the_analytic_tier(channel):
    _, eb = channel
    with pytest.raises(ValueError, match="walled channel"):
        sw.single_wave(eb, {"x": 2, "y": 1})


# ================================================================
#  jet (the Jet port)
# ================================================================
def test_geostrophic_jet_is_steady_in_the_linear_model(periodic):
    model, em = periodic
    z = sw.jet(em)
    tendency = model.tendency(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in COMPONENTS)
    residual = max(
        float(np.abs(np.asarray(tendency[c].data)).max())
        for c in COMPONENTS)
    assert scale > 0.5
    assert residual < 1e-12 * scale


def test_jet_profile_peaks_where_asked(periodic):
    _, em = periodic
    z = sw.jet(em, pos=0.25, width=0.1, waveamp=0.0,
               geo_proj=False)
    u = np.asarray(z["u"].data)
    assert float(np.abs(u).max()) == pytest.approx(1.0)
    nodes = np.asarray(em.grid.evaluation_nodes(
        z["u"].function_space, "y").data).ravel()
    peak = nodes[np.abs(u).max(axis=0).argmax()]
    assert abs(peak - 0.25) <= 0.5 / N + 1e-12
    # the perturbation rides on top of the normalized jet
    zp = sw.jet(em, pos=0.25, width=0.1, waveamp=0.1,
                geo_proj=False)
    assert not _same(z, zp)


# ================================================================
#  coherent_eddy (the CoherentEddy port)
# ================================================================
@pytest.mark.parametrize("gauss_field",
                         ["vorticity", "streamfunction"])
def test_eddy_is_divergence_free_and_balanced(periodic, gauss_field):
    model, em = periodic
    z = sw.coherent_eddy(em, width=0.2, gauss_field=gauss_field)
    div = float(np.abs(np.asarray(z.divergence.data)).max())
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    assert div < 1e-12 * umax
    # p = f0 psi balances the velocities to discretization accuracy
    tendency = model.tendency(z)
    residual = max(
        float(np.abs(np.asarray(tendency[c].data)).max())
        for c in ("u", "v"))
    assert residual < 0.15 * em.f0 * umax


def test_eddy_streamfunction_centers_the_pressure(periodic):
    _, em = periodic
    z = sw.coherent_eddy(em, pos_x=0.25, pos_y=0.75, width=0.15,
                         gauss_field="streamfunction")
    p = np.asarray(z["p"].data)
    ix, iy = np.unravel_index(np.abs(p).argmax(), p.shape)
    grid = em.grid
    xs = np.asarray(grid.evaluation_nodes(
        z["p"].function_space, "x").data).ravel()
    ys = np.asarray(grid.evaluation_nodes(
        z["p"].function_space, "y").data).ravel()
    assert abs(xs[ix] - 0.25) <= 0.5 / N + 1e-12
    assert abs(ys[iy] - 0.75) <= 0.5 / N + 1e-12
    # a negative amplitude flips the rotation sense exactly
    flipped = sw.coherent_eddy(em, pos_x=0.25, pos_y=0.75,
                               width=0.15, amplitude=-1.0,
                               gauss_field="streamfunction")
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(flipped[c].data),
                              -np.asarray(z[c].data))


def test_eddy_taught_errors(periodic, channel):
    _, em = periodic
    with pytest.raises(ValueError, match="unknown gauss_field"):
        sw.coherent_eddy(em, gauss_field="pressure")
    _, eb = channel
    with pytest.raises(ValueError, match="walled channel"):
        sw.coherent_eddy(eb)
    with pytest.raises(ValueError, match="walled channel"):
        sw.jet(eb)


# ================================================================
#  equatorial_wave (the EquatorialWave port)
# ================================================================
BETA = 8.0
N_EQ = 32


def _beta_model(grid, beta=BETA):
    """Build a linear beta-plane model (equator mid-domain)."""
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=-2.0 * beta,
                                              beta=beta),
        advection=False,
        time_stepper=fr.time_steppers.AdamBashforth(1e-3, order=3))


@pytest.fixture(scope="module")
def equatorial():
    """One linear beta-plane model with the equator mid-domain."""
    mx = fr.grid.meshes.IntervalMesh(N_EQ, (0.0, 4.0),
                                     periodic=True, name="x")
    my = fr.grid.meshes.IntervalMesh(N_EQ, (0.0, 4.0),
                                     periodic=True, name="y")
    return _beta_model(fr.grid.Grid((mx, my)))


def test_equatorial_wave_satisfies_the_eigen_relation(equatorial):
    # d/dt state(phase) ~ omega * state(phase + pi/2) through the
    # real linear tendency, to discretization accuracy
    omega, z0 = sw.equatorial_wave(equatorial, 2, 1, 2, phase=0.4)
    _, z1 = sw.equatorial_wave(equatorial, 2, 1, 2,
                               phase=0.4 + np.pi / 2)
    tau = equatorial.tendency(z0)
    num = max(
        float(np.abs(np.asarray(tau[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in COMPONENTS)
    den = abs(omega) * max(
        float(np.abs(np.asarray(z1[c].data)).max())
        for c in COMPONENTS)
    assert num / den < 0.05


def test_equatorial_wave_traps_at_the_equator(equatorial):
    _omega, z = sw.equatorial_wave(equatorial, 2, 0, 2)
    v = np.asarray(z["v"].data)
    ys = np.asarray(equatorial.grid.evaluation_nodes(
        z["v"].function_space, "y").data).ravel()
    peak = ys[np.abs(v).max(axis=0).argmax()]
    assert abs(peak - 2.0) <= 4.0 / N_EQ + 1e-12
    edge = np.abs(v[:, [0, -1]]).max()
    assert edge < 1e-3 * np.abs(v).max()
    # normalization: the largest horizontal velocity is one
    umax = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    assert umax == pytest.approx(1.0, abs=1e-12)
    # the equator override recentres the trapping latitude
    _, shifted = sw.equatorial_wave(equatorial, 2, 0, 2,
                                    equator=1.0)
    vs = np.asarray(shifted["v"].data)
    peak = ys[np.abs(vs).max(axis=0).argmax()]
    assert abs(peak - 1.0) <= 4.0 / N_EQ + 1e-12


def test_equatorial_wave_orders_the_dispersion_roots(equatorial):
    omegas = [
        sw.equatorial_wave(equatorial, 2, 1, mode)[0]
        for mode in (0, 1, 2)]
    assert omegas[0] < omegas[1] < omegas[2]
    assert omegas[0] < 0 < omegas[2]
    # the middle root is the slow Rossby wave
    assert abs(omegas[1]) < min(abs(omegas[0]), abs(omegas[2]))


def test_equatorial_wave_taught_errors(equatorial, periodic):
    model, _ = periodic
    with pytest.raises(ValueError, match="beta plane"):
        sw.equatorial_wave(model, 2, 1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        sw.equatorial_wave(equatorial, 2, -1, 2)
    with pytest.raises(ValueError, match="wave_mode"):
        sw.equatorial_wave(equatorial, 2, 1, 3)
    walled_x = _beta_model(make_grid(periodic_x=False))
    with pytest.raises(ValueError, match="periodic zonal axis"):
        sw.equatorial_wave(walled_x, 2, 1, 2)
    negative = _beta_model(make_grid(), beta=-1.0)
    with pytest.raises(ValueError, match="beta > 0"):
        sw.equatorial_wave(negative, 2, 1, 2)


# ================================================================
#  Multi-device: same seed, any device count
# ================================================================
@pytest.mark.multi_device
def test_random_state_is_device_count_invariant(forced_devices):
    # the phases are keyed on global DOF indices and the transforms
    # compose under the decomposition, so the same seed realizes
    # the same state on any device layout
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        mx = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0),
                                         periodic=True, name="x")
        my = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0),
                                         periodic=True, name="y")
        model = make_model(
            fr.grid.Grid((mx, my), device_ids=device_ids),
            csqr=CSQR, f0=1.5, advection=False)
        em = sw.eigenmodes.from_model(model)
        results[tag] = sw.random_vortical(em, seed=21)
        if tag == "many":
            data = results[tag]["u"]._data
            assert len(data.sharding.device_set) == jax.device_count()
    assert max(
        float(np.abs(np.asarray(results["many"][c].data)
                     - np.asarray(results["one"][c].data)).max())
        for c in COMPONENTS) < 1e-12
