"""The shared eigenmode initial-condition primitives (eigenstates).

Covers the Masur & Oliver geostrophic spectrum, the axis-keyed
coefficient-index resolution (Fourier half/full layouts, the trig
union lattice), the Hermitian single-mode placement, and the
amplitude-convention helpers. The synthesis pipelines built on these
(``em.mode`` / ``eb.mode`` / the random states) are exercised end to
end by the package suites and ``test_eigenbasis``.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.symbols import GridSymbols
from fridom.model.eigenstates import (
    coefficient_index,
    envelope_scale,
    geostrophic_energy_spectrum,
    hermitian_mode_data,
    normalize_max_component,
)

N = 16


@pytest.fixture
def spectra_space():
    """Return the rfft space: half-spectrum x, full-complex y."""
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    return (mx.fourier(origin=mx.center)
            * my.fourier(origin=my.center.as_complex()))


@pytest.fixture
def trig_space():
    """Half-spectrum x times the DST-I (Dirichlet inner-face) z."""
    mx = IntervalMesh(N, (0.0, 1.0), periodic=True, name="x")
    mz = IntervalMesh(8, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, mz))
    space = fr.spatial.Staggered(
        "z", wall_bc={"z": BC.DIRICHLET}).resolve(grid)
    return GridSymbols(grid, {"w": space}).coeff("w")


# ================================================================
#  The geostrophic energy spectrum
# ================================================================
def test_geostrophic_spectrum_peaks_near_k0_and_vanishes_at_zero():
    k = np.linspace(0.0, 40.0, 4001)
    s = np.asarray(geostrophic_energy_spectrum(k, 0.0, k0=2.0))
    assert s[0] == 0.0
    assert abs(k[int(np.argmax(s))] - 2.0) < 0.2


def test_geostrophic_spectrum_large_k_power_law():
    # S ~ k^-d for k -> infinity: the log-log slope approaches -d
    d = 5.0
    k1, k2 = 200.0, 400.0
    s1 = float(geostrophic_energy_spectrum(k1, 0.0, d=d))
    s2 = float(geostrophic_energy_spectrum(k2, 0.0, d=d))
    slope = np.log(s2 / s1) / np.log(k2 / k1)
    assert abs(slope + d) < 0.05


# ================================================================
#  Axis-keyed coefficient indices
# ================================================================
def test_coefficient_index_resolves_half_and_full_axes(
        spectra_space):
    assert coefficient_index(
        spectra_space, {"x": 3, "y": 2}) == (3, 2)
    # full-spectrum axes take any integer modulo n
    assert coefficient_index(
        spectra_space, {"x": 0, "y": -2}) == (0, N - 2)


def test_coefficient_index_rejects_wrong_keys(spectra_space):
    with pytest.raises(ValueError, match="keyed by the grid axes"):
        coefficient_index(spectra_space, {"x": 1})


def test_coefficient_index_rejects_out_of_half_range(spectra_space):
    with pytest.raises(ValueError, match="half"):
        coefficient_index(spectra_space, {"x": -1, "y": 0})
    with pytest.raises(ValueError, match="half"):
        coefficient_index(spectra_space, {"x": N // 2 + 1, "y": 0})


def test_coefficient_index_trig_union_modes(trig_space):
    # DST-I holds modes 1..n-1 at slots 0..n-2
    assert coefficient_index(trig_space, {"x": 2, "z": 3}) == (2, 2)
    # union modes the family lacks resolve to None, never an error
    assert coefficient_index(trig_space, {"x": 2, "z": 0}) is None
    assert coefficient_index(trig_space, {"x": 2, "z": 8}) is None


def test_coefficient_index_rejects_out_of_union_range(trig_space):
    with pytest.raises(ValueError, match="union modes"):
        coefficient_index(trig_space, {"x": 2, "z": 9})


def test_coefficient_index_rejects_non_coefficient_spaces():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    with pytest.raises(ValueError, match="not a coefficient"):
        coefficient_index(mx.center * my.center, {"x": 1, "y": 0})


# ================================================================
#  Hermitian single-mode placement
# ================================================================
def test_hermitian_mode_data_interior_is_a_single_entry(
        spectra_space):
    value = 0.3 + 0.4j
    data = hermitian_mode_data(spectra_space, (3, 2), value)
    assert data.shape == (N // 2 + 1, N)
    assert data[3, 2] == value
    assert float(jnp.abs(data).sum()) == pytest.approx(abs(value))


def test_hermitian_mode_data_closes_self_conjugate_planes(
        spectra_space):
    value = 0.3 + 0.4j
    data = hermitian_mode_data(spectra_space, (0, 2), value)
    assert data[0, 2] == value / 2
    assert data[0, N - 2] == jnp.conj(value) / 2
    # fully self-conjugate index: the pair degenerates to Re(value)
    data = hermitian_mode_data(spectra_space, (N // 2, 0), value)
    assert data[N // 2, 0] == value.real


def test_hermitian_mode_data_needs_complex_storage():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    with pytest.raises(ValueError, match="complex storage"):
        hermitian_mode_data(mx.center * my.center, (0, 0), 1.0)


# ================================================================
#  Amplitude conventions
# ================================================================
class _Field:

    """A minimal .data carrier standing in for a ScalarField."""

    def __init__(self, data):
        self.data = jnp.asarray(data)

    def __truediv__(self, scale):
        return _Field(self.data / scale)


def test_envelope_scale_is_the_peak_quadrature_envelope():
    z0 = {"u": _Field([3.0, 0.0]), "v": _Field([0.0, 0.0])}
    z1 = {"u": _Field([4.0, 0.0]), "v": _Field([1.0, 0.0])}
    assert envelope_scale(z0, z1, ("u", "v")) == pytest.approx(5.0)


def test_envelope_scale_zero_falls_back_to_one():
    z = {"u": _Field([0.0]), "v": _Field([0.0])}
    assert envelope_scale(z, z, ("u", "v")) == 1.0


def test_normalize_max_component_scales_all_components():
    fields = {"u": _Field([0.5, -2.0]), "v": _Field([1.0, 0.0]),
              "p": _Field([4.0, 4.0])}
    out = normalize_max_component(fields, ("u", "v"))
    assert float(jnp.abs(out["u"].data).max()) == pytest.approx(1.0)
    assert float(out["p"].data.max()) == pytest.approx(2.0)


def test_normalize_max_component_zero_velocity_is_untouched():
    fields = {"u": _Field([0.0]), "v": _Field([0.0]),
              "p": _Field([3.0])}
    out = normalize_max_component(fields, ("u", "v"))
    assert float(out["p"].data.max()) == 3.0
