"""Tests for fridom.framework2.grid.random_fields."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.random_fields import (
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


def test_phase_not_implemented(grid, mx, my):
    with pytest.raises(NotImplementedError, match="phase"):
        grid.random.phase(mx.center * my.center, seed=0)


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
