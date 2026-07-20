"""Initial conditions (sw.initial_conditions).

Validates the prescribed-spectra random states on both eigenmode
tiers: family purity (the projections recover ~100% of the energy),
the realized per-mode energy against the prescribed ``S/(pi k_h)``
convention (deterministic amplitudes: unit-modulus phases), bitwise
seed determinism, the normalization convention, the taught errors,
and the multi-device device-count invariance of the analytic path.

The named analytic ports: single_wave phase-rotates exactly in the
linear model, the geostrophically projected jet is steady, and the
coherent eddy is discretely divergence-free and balanced. Beta-plane
wave modes are selected numerically through the channel eigenbasis
(tests in test_channel_eigenmodes.py).
"""
from itertools import pairwise

import jax
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.eigenstates import (
    geostrophic_energy_spectrum,
)

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")
CSQR = 2.0


def _one_device_grid(*, periodic_y=True):
    # device_ids=(0,) twin of the conftest make_grid: pins the shared
    # module fixtures to one device so the math runs at any device
    # count, and serves as the reference build for the device-count
    # invariance tests (the sharded paths are served by the fused
    # distributed routes and compared against this twin).
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my), device_ids=(0,))


@pytest.fixture(scope="module")
def periodic():
    """One linear periodic model + analytic eigenmodes (shared)."""
    model = make_model(_one_device_grid(), csqr=CSQR, f0=1.5,
                       advection=False)
    return model, sw.eigenmodes.from_model(model)


@pytest.fixture(scope="module")
def channel():
    """One walled channel model + labeled eigenbasis (shared)."""
    model = make_model(_one_device_grid(periodic_y=False), csqr=CSQR,
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


@pytest.mark.multi_device
def test_channel_random_state_on_a_sharded_grid_is_device_invariant(
        forced_devices):
    # WAS a taught error: the channel random-state synthesis routes
    # through the fused channel contraction (Channel2DPlan) since the
    # distributed-transform campaign, so a grid that shards the
    # periodic axis reproduces the one-device reference.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    model = make_model(make_grid(periodic_y=False), csqr=CSQR,
                       f0=1.5, advection=False)
    sharded = sw.random_vortical(sw.eigenbasis(model), seed=4)
    reference = make_model(_one_device_grid(periodic_y=False),
                           csqr=CSQR, f0=1.5, advection=False)
    expected = sw.random_vortical(sw.eigenbasis(reference), seed=4)
    for c in COMPONENTS:
        assert np.allclose(np.asarray(sharded[c].data),
                           np.asarray(expected[c].data),
                           rtol=0.0, atol=1e-12)


# ================================================================
#  single_wave (the SingleWave port)
# ================================================================
@pytest.fixture(scope="module")
def wave_setup():
    """One linear periodic model with a wave-resolving time step."""
    model = make_model(_one_device_grid(), csqr=CSQR, f0=1.5,
                       advection=False, dt=1e-3)
    return model, sw.eigenmodes.from_model(model)


def test_single_wave_phase_rotates_in_the_linear_model(wave_setup):
    model, em = wave_setup
    omega, z0 = sw.single_wave(em, {"x": 2, "y": 1}, "wave+",
                               phase=0.3)
    model.set_state(z0)
    steps = 20
    model.advance(steps)
    _, zt = sw.single_wave(em, {"x": 2, "y": 1}, "wave+",
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
    omega, z = sw.single_wave(model, {"x": 3, "y": 2}, "wave",
                              branch=-1,
                              phase=0.7)
    omega_em, z_em = em.mode("wave-", {"x": 3, "y": 2}, phase=0.7)
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
#  Multi-device: same seed, any device count
# ================================================================
def test_random_state_is_deterministic_on_one_device():
    # the single-device math: the same seed realizes the same state, a
    # different seed a different one (device-count invariance is dead
    # test debt now the sharded path is a Tier-1 taught error, asserted
    # below). device_ids=(0,) keeps this valid at any device count.
    model = make_model(_one_device_grid(), csqr=CSQR, f0=1.5,
                       advection=False)
    em = sw.eigenmodes.from_model(model)
    assert _same(sw.random_vortical(em, seed=21),
                 sw.random_vortical(em, seed=21))
    assert not _same(sw.random_vortical(em, seed=21),
                     sw.random_vortical(em, seed=22))


def _sharded_pair(device_ids):
    """Return the (sharded, one-device) 2-D periodic eigenmodes pair."""
    def build(ids):
        mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                         periodic=True, name="x")
        my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                         periodic=True, name="y")
        model = make_model(
            fr.spatial.Grid((mx, my), device_ids=ids),
            csqr=CSQR, f0=1.5, advection=False)
        return sw.eigenmodes.from_model(model)
    return build(device_ids), build((0,))


@pytest.mark.multi_device
def test_random_state_on_a_sharded_grid_is_device_invariant(
        forced_devices):
    # WAS a taught error: the analytic random-state synthesis builds its
    # gains and Hermitian random phases on the device-independent
    # single-device coefficient frame (grid.random keys on the global
    # storage index, deterministic across device counts). The 2-D layout
    # shards the half axis, so the fused transpose re-designates it to the
    # full complex spectrum and the synthesis routes through the fused
    # jax.shard_map backward via the Hermitian half-axis re-expression (no
    # gather) -- reproducing the one-device state to floating point, real
    # and sharded, on any device count.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = _sharded_pair(None)
    for family in ("vortical", "wave"):
        s_many = sw.random_state(many, family, seed=21)
        s_one = sw.random_state(one, family, seed=21)
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
    # half-axis re-expression on the 2-D sharded grid, so the sharded mode
    # reproduces the one-device mode (frequency and field) to floating
    # point, real and sharded.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = _sharded_pair(None)
    k = {"x": 3, "y": 2}
    om_many, z_many = sw.single_wave(many, k, "wave+", phase=0.4)
    om_one, z_one = sw.single_wave(one, k, "wave+", phase=0.4)
    assert om_many == om_one
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    assert all(not np.iscomplexobj(np.asarray(z_many[c].data))
               for c in COMPONENTS)
    err = max(
        float(np.abs(np.asarray(z_many[c].data)
                     - np.asarray(z_one[c].data)).max())
        for c in COMPONENTS)
    assert err < 1e-11
