"""Tests for fridom.framework2.grid.fields.storage."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.fields.storage import (
    factor_axes,
    hermitian_project,
    self_conjugate_axis_indices,
    storage_dtype,
    store,
)
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 2.0), periodic=False, name="y")


def test_storage_dtype_real_nodal(mx):
    assert storage_dtype(mx.center) == jnp.float64


def test_storage_dtype_complex_nodal(mx):
    assert storage_dtype(mx.center.as_complex()) == jnp.complex128


def test_storage_dtype_real_origin_fourier_is_complex(mx):
    space = mx.fourier(origin=mx.center)
    assert storage_dtype(space) == jnp.complex128


def test_storage_dtype_sine_coefficients_stay_real(my):
    origin = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert storage_dtype(my.sine(origin)) == jnp.float64


def test_storage_dtype_product_promotes(mx, my):
    space = mx.center * my.center.as_complex()
    assert storage_dtype(space) == jnp.complex128


def test_factor_axes(mx, my):
    space = mx.center * my.center
    assert factor_axes(space) == ((mx.center, 0), (my.center, 1))


def test_self_conjugate_indices_even_origin(mx):
    factor = mx.fourier(origin=mx.center)
    # n = 8: half spectrum of length 5, Nyquist at index 4
    assert self_conjugate_axis_indices(factor) == (0, 4)


def test_self_conjugate_indices_odd_origin():
    mesh = IntervalMesh(9, (0.0, 1.0), name="x")
    factor = mesh.fourier(origin=mesh.center)
    assert self_conjugate_axis_indices(factor) == (0,)


def test_hermitian_project_zeroes_self_conjugate_imag(mx):
    space = mx.fourier(origin=mx.center)
    data = jnp.full(space.shape, 1.0 + 1.0j)
    out = hermitian_project(data, space)
    assert out[0] == 1.0 + 0.0j
    assert out[4] == 1.0 + 0.0j
    assert jnp.array_equal(out[1:4], data[1:4])


def test_hermitian_project_product_masks_per_axis(mx, my):
    space = mx.fourier(origin=mx.center) * my.center
    data = jnp.full(space.shape, 2.0 + 3.0j)
    out = hermitian_project(data, space)
    assert jnp.array_equal(out[0], jnp.full(4, 2.0 + 0.0j))
    assert jnp.array_equal(out[4], jnp.full(4, 2.0 + 0.0j))
    assert jnp.array_equal(out[1:4], data[1:4])


def test_hermitian_project_noop_on_nodal(mx):
    data = jnp.arange(8.0)
    assert hermitian_project(data, mx.center) is data


def test_hermitian_project_noop_on_complex_origin_fourier(mx):
    space = mx.fourier(origin=mx.center.as_complex())
    data = jnp.full(space.shape, 1.0 + 1.0j)
    assert hermitian_project(data, space) is data


def test_store_roundtrips_through_unpad(mx, my):
    dec = TensorDecomposition(
        meshes=(mx, my), names=("x", "y"),
        halo=HaloSpec.zero(("x", "y")), layouts=(Layout({}),))
    space = mx.center * my.center
    data = jnp.arange(32.0).reshape(8, 4)
    stored = store(dec, space, data)
    assert jnp.array_equal(dec.unpad(stored, space), data)
