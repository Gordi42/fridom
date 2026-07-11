"""Tests for fridom.spatial.random_fields."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.random_fields import (
    RandomFieldFactory,
    _global_indices,
)


@pytest.fixture
def mx():
    return IntervalMesh(16, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(16, (0.0, 2.0), name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


def test_random_property_is_grid_bound(grid):
    assert isinstance(grid.random, RandomFieldFactory)
    assert grid.random is grid.random


def test_normal_returns_field_on_laid_out_space(grid, mx, my):
    f = grid.random.normal(mx.center * my.center, seed=0)
    assert f.grid is grid
    assert f.function_space.bare is mx.center * my.center
    assert f.function_space.layout is grid.decomposition.default_layout
    assert f.shape == (16, 16)
    assert f.dtype == jnp.float64


def test_normal_deterministic_per_seed(grid, mx, my):
    space = mx.center * my.center
    a = grid.random.normal(space, seed=7)
    b = grid.random.normal(space, seed=7)
    c = grid.random.normal(space, seed=8)
    assert jnp.array_equal(a.data, b.data)
    assert not jnp.array_equal(a.data, c.data)


def test_normal_moments(grid, mx, my):
    f = grid.random.normal(mx.center * my.center, seed=3)
    assert abs(float(f.data.mean())) < 0.2
    assert abs(float(f.data.std()) - 1.0) < 0.15


def test_normal_keyed_by_global_true_dof_index(grid, mx, my):
    # the draw is a pure function of (space.shape, seed) over the
    # global flat index: recompute one DOF's value independently
    f = grid.random.normal(mx.center * my.center, seed=5)
    key = jax.random.key(5)
    flat_index = 3 * 16 + 11  # row-major index of DOF (3, 11)
    expected = jax.random.normal(
        jax.random.fold_in(key, flat_index), (1,))[0]
    assert jnp.allclose(f.data[3, 11], expected)


def test_normal_complex_on_complex_space(grid, mx, my):
    space = (mx.center * my.center).as_complex()
    f = grid.random.normal(space, seed=1)
    assert f.dtype == jnp.complex128
    assert float(jnp.abs(f.data.imag).max()) > 0.0
    # complex normal with unit total variance
    assert abs(float((jnp.abs(f.data) ** 2).mean()) - 1.0) < 0.2


def test_normal_hermitian_on_real_origin_fourier(mx):
    grid = Grid((mx,))
    space = mx.fourier(origin=mx.center)
    f = grid.random.normal(space, seed=2)
    assert f.dtype == jnp.complex128
    assert f.shape == (9,)  # n = 16: half spectrum
    # self-conjugate modes (k = 0 and Nyquist) are drawn real
    assert f.data[0].imag == 0.0
    assert f.data[-1].imag == 0.0
    assert float(jnp.abs(f.data[1:-1].imag).min()) > 0.0


def test_normal_hermitian_product_with_nodal_factor(mx, my):
    # the half-spectrum factor is the sole complex carrier: the
    # flat real-draw rule applies on the whole k = 0/Nyquist planes
    grid = Grid((mx, my))
    space = mx.fourier(origin=mx.center) * my.center
    f = grid.random.normal(space, seed=3)
    assert jnp.abs(f.data[0].imag).max() == 0.0
    assert jnp.abs(f.data[-1].imag).max() == 0.0
    assert float(jnp.abs(f.data[1:-1].imag).min()) > 0.0


def test_normal_multi_axis_spectra_have_no_flat_real_plane():
    # shared guard with the field factory (flat_hermitian_applies):
    # with a second complex-carrying factor the invariant is the
    # conjugate pairing, so no plane of forced-real draws applies
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    mp = IntervalMesh(4, (0.0, 2.0), name="p")
    grid = Grid((mx, mp))
    space = (mx.fourier(origin=mx.center)
             * mp.fourier(origin=mp.center.as_complex()))
    f = grid.random.normal(space, seed=4)
    assert float(jnp.abs(f.data[0].imag).min()) > 0.0


def test_normal_no_nyquist_on_odd_origin():
    mesh = IntervalMesh(9, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    f = grid.random.normal(mesh.fourier(origin=mesh.center), seed=2)
    assert f.shape == (5,)
    assert f.data[0].imag == 0.0
    assert float(jnp.abs(f.data[1:].imag).min()) > 0.0


def test_normal_is_jit_traceable(grid, mx, my):
    space = mx.center * my.center

    @jax.jit
    def draw():
        return grid.random.normal(space, seed=4).data

    assert jnp.array_equal(draw(),
                           grid.random.normal(space, seed=4).data)


# ================================================================
#  phase: Hermitian unit-modulus random phases
# ================================================================
@pytest.fixture
def spectra_space(mx, my):
    """Return the rfft coefficient space (half x, full-complex y)."""
    return (mx.fourier(origin=mx.center)
            * my.fourier(origin=my.center.as_complex()))


def test_phase_unit_modulus_everywhere(grid, spectra_space):
    f = grid.random.phase(spectra_space, seed=9)
    assert f.dtype == jnp.complex128
    assert float(jnp.abs(jnp.abs(f.data) - 1.0).max()) < 1e-14


def test_phase_pairing_is_exact_on_self_conjugate_planes(
        grid, spectra_space):
    # on the kx = 0 / Nyquist planes the conjugate pairing across the
    # full-spectrum y axis is bitwise (shared canonical key + conj)
    d = grid.random.phase(spectra_space, seed=9).data
    mirror = (-jnp.arange(16)) % 16
    for plane in (0, -1):
        assert jnp.array_equal(d[plane], jnp.conj(d[plane][mirror]))


def test_phase_fully_self_conjugate_dofs_are_real_signs(
        grid, spectra_space):
    d = grid.random.phase(spectra_space, seed=9).data
    corners = jnp.stack(
        [d[0, 0], d[0, 8], d[-1, 0], d[-1, 8]])
    assert jnp.abs(corners.imag).max() == 0.0
    assert jnp.array_equal(jnp.abs(corners.real), jnp.ones(4))


def test_phase_interior_dofs_are_free(grid, spectra_space):
    d = grid.random.phase(spectra_space, seed=9).data
    assert float(jnp.abs(d[1:-1].imag).min()) > 0.0


def test_phase_backward_transform_is_exactly_real(
        grid, spectra_space):
    # the full-spectrum reconstruction (explicit conjugate half) has
    # a vanishing imaginary part: the field is a valid rfftn spectrum
    d = np.asarray(grid.random.phase(spectra_space, seed=4).data)
    full = np.zeros((16, 16), dtype=complex)
    full[:9] = d
    mirror = (-np.arange(16)) % 16
    full[9:] = np.conj(d[1:8][::-1][:, mirror])
    phys = np.fft.ifft2(full)
    assert np.abs(phys.imag).max() < 1e-15


def test_phase_deterministic_per_seed(grid, spectra_space):
    a = grid.random.phase(spectra_space, seed=7)
    b = grid.random.phase(spectra_space, seed=7)
    c = grid.random.phase(spectra_space, seed=8)
    assert jnp.array_equal(a.data, b.data)
    assert not jnp.array_equal(a.data, c.data)


def test_phase_1d_half_spectrum(mx):
    grid = Grid((mx,))
    f = grid.random.phase(mx.fourier(origin=mx.center), seed=2)
    assert f.data[0].imag == 0.0
    assert f.data[-1].imag == 0.0
    assert float(jnp.abs(f.data[1:-1].imag).min()) > 0.0


def test_phase_complex_space_without_fourier_factor_is_free(
        grid, mx, my):
    # complex storage with no half-spectrum factor carries no
    # Hermitian constraint: every DOF is a free unit phase
    f = grid.random.phase((mx.center * my.center).as_complex(),
                          seed=6)
    assert f.dtype == jnp.complex128
    assert float(jnp.abs(jnp.abs(f.data) - 1.0).max()) < 1e-14
    assert float(jnp.abs(f.data.imag).min()) > 0.0


def test_phase_real_storage_draws_signs(grid, mx, my):
    # a real-storage space has no Fourier factor: unit magnitude
    # means Rademacher +1/-1 per DOF
    f = grid.random.phase(mx.center * my.center, seed=3)
    assert f.dtype == jnp.float64
    assert jnp.array_equal(jnp.abs(f.data), jnp.ones((16, 16)))
    assert float(f.data.min()) == -1.0


@pytest.mark.multi_device
def test_phase_is_device_count_invariant(forced_devices):
    # per-DOF keying over the global true index: the same seed gives
    # bitwise identical draws on any device layout
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        mesh_x = IntervalMesh(16, (0.0, 1.0), name="x")
        mesh_y = IntervalMesh(16, (0.0, 2.0), name="y")
        grid = Grid((mesh_x, mesh_y), device_ids=device_ids)
        space = (mesh_x.fourier(origin=mesh_x.center)
                 * mesh_y.fourier(
                     origin=mesh_y.center.as_complex()))
        results[tag] = np.asarray(
            grid.random.phase(space, seed=11).data)
    assert np.array_equal(results["many"], results["one"])


def test_global_indices_cover_the_full_true_shape():
    shape = (3, 4)
    slices = (slice(0, 3), slice(0, 4))
    idx = _global_indices(shape, slices)
    assert jnp.array_equal(idx, jnp.arange(12).reshape(3, 4))


def test_global_indices_of_a_subrange():
    shape = (3, 4)
    slices = (slice(1, 3), slice(2, 4))
    idx = _global_indices(shape, slices)
    expected = jnp.arange(12).reshape(3, 4)[1:3, 2:4]
    assert jnp.array_equal(idx, expected)
