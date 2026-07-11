"""Tests for fridom.spatial.fields.storage."""
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.spatial.fields.storage import (
    factor_axes,
    flat_hermitian_applies,
    hermitian_project,
    self_conjugate_axis_indices,
    storage_dtype,
    store,
)
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodeSet


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


def test_flat_hermitian_guard(mx, my):
    mp = IntervalMesh(4, (0.0, 2.0), name="p")  # periodic
    half_x = mx.fourier(origin=mx.center)
    full_p = mp.fourier(origin=mp.center.as_complex())
    assert flat_hermitian_applies(half_x)
    assert flat_hermitian_applies(half_x * my.center)
    assert not flat_hermitian_applies(mx.center * my.center)
    assert not flat_hermitian_applies(full_p)  # no half factor
    # half + a second complex-carrying factor: pairing, not flat
    assert not flat_hermitian_applies(half_x * full_p)
    assert not flat_hermitian_applies(
        half_x * my.center.as_complex())


def test_hermitian_project_preserves_multi_axis_spectra(mx):
    # regression (wave-3 merge): on Fourier(x, half) x Fourier(p,
    # full) the invariant is the conjugate *pairing* on the k=0 and
    # Nyquist planes of the halved axis; flat imag-zeroing would
    # corrupt a perfectly valid rfftn spectrum. Valid data must pass
    # through unmodified.
    mp = IntervalMesh(4, (0.0, 2.0), name="p")
    space = (mx.fourier(origin=mx.center)
             * mp.fourier(origin=mp.center.as_complex()))
    values = jnp.arange(32.0).reshape(8, 4)
    spectrum = jnp.fft.fft(
        jnp.fft.rfft(values, axis=0, norm="forward"),
        axis=1, norm="forward")
    assert spectrum.shape == space.shape
    assert hermitian_project(spectrum, space) is spectrum


def test_store_roundtrips_through_unpad(mx, my):
    dec = TensorDecomposition(
        meshes=(mx, my), names=("x", "y"),
        halo=HaloSpec.zero(("x", "y")), layouts=(Layout({}),))
    space = mx.center * my.center
    data = jnp.arange(32.0).reshape(8, 4)
    stored = store(dec, space, data)
    assert jnp.array_equal(dec.unpad(stored, space), data)
