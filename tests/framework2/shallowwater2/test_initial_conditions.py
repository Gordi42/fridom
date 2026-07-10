"""Random-phase eigenmode initial conditions (sw.initial_conditions).

Validates the prescribed-spectra random states on both eigenmode
tiers: family purity (the projections recover ~100% of the energy),
the realized per-mode energy against the prescribed ``S/(pi k_h)``
convention (deterministic amplitudes: unit-modulus phases), bitwise
seed determinism, the normalization convention, the taught errors,
and the multi-device device-count invariance of the analytic path.
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
    # the interpolation-Nyquist planes are structural zeros of the
    # geostrophic family (no vortical mode to populate); the far
    # spectral tail is excluded against fp leakage
    represented = np.ones_like(expected, dtype=bool)
    represented[N // 2, :] = False
    represented[:, N // 2] = False
    mask = ((kh > 0) & represented
            & (expected > 1e-6 * expected.max()))
    ratio = energy[mask] / expected[mask]
    assert ratio.max() / ratio.min() < 1.0 + 1e-6
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
