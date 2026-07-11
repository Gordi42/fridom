"""
Wave-1 cross-cluster integration: real spaces through the decomposition.

Description
-----------
The Wave-1 clusters were built in parallel — meshes/spaces against the
class-doc signatures, the decomposition against a duck-typed space
protocol. These tests exercise the real seams: mesh-minted interned
spaces flowing through ``TensorDecomposition``, the layout protocol
against the real ``Layout`` value, and the ``decomposition_traits``
seam on real spaces.
"""

import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.decomposition import SpaceLike
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import TensorDecomposition
from fridom.framework2.grid.decomposition.traits import HaloStrategy
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.tensor_product import TensorProductSpace


@pytest.fixture
def meshes():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(4, (0.0, 2.0), name="y")
    return mx, my


@pytest.fixture
def decomp(meshes):
    mx, my = meshes
    layout = Layout({})
    return TensorDecomposition(
        meshes=(mx, my),
        names=("x", "y"),
        halo=HaloSpec.zero(("x", "y")),
        layouts=(layout,),
    ), layout


def test_real_product_satisfies_space_protocol(meshes):
    mx, my = meshes
    space = mx.center * my.center
    assert isinstance(space, SpaceLike)
    assert isinstance(mx.center, SpaceLike)


def test_storage_shape_matches_true_shape_at_halo_zero(meshes, decomp):
    mx, my = meshes
    dec, _ = decomp
    space = mx.center * my.center
    assert dec.storage_shape(space) == space.shape == (8, 4)


def test_zeros_pad_sync_unpad_roundtrip_on_real_space(meshes, decomp):
    mx, my = meshes
    dec, _ = decomp
    space = mx.center * my.center
    z = dec.zeros(space)
    assert z.shape == dec.storage_shape(space)
    data = jnp.arange(32.0).reshape(8, 4)
    stored = dec.sync(dec.pad(data, space), space)
    assert jnp.array_equal(dec.unpad(stored, space), data)


def test_with_layout_interns_against_real_layout(meshes, decomp):
    mx, my = meshes
    _, layout = decomp
    bare = mx.center * my.center
    laid = bare.with_layout(layout)
    assert laid is bare.with_layout(layout)
    assert laid is not bare
    assert laid.bare is bare
    assert laid.layout is layout
    # per-factor access stays bare (dispatch keys never see layouts)
    assert laid.factor("x") is mx.center


def test_laid_out_space_through_decomposition(meshes, decomp):
    mx, my = meshes
    dec, layout = decomp
    laid = (mx.center * my.center).with_layout(layout)
    assert dec.storage_shape(laid) == (8, 4)
    data = jnp.ones((8, 4))
    assert jnp.array_equal(dec.unpad(dec.pad(data, laid), laid), data)


def test_traits_seam_on_real_spaces(meshes):
    mx, _ = meshes
    nodal = mx.decomposition_traits(mx.center)
    assert nodal.strategies[0] is HaloStrategy.GHOST
    coeff = mx.decomposition_traits(mx.fourier(origin=mx.center))
    assert HaloStrategy.LOCAL in coeff.strategies
    const = mx.decomposition_traits(mx.constant)
    assert const.strategies == (HaloStrategy.LOCAL,)


def test_single_factor_space_through_decomposition(meshes):
    mx, _ = meshes
    dec = TensorDecomposition(
        meshes=(mx,),
        names=("x",),
        halo=HaloSpec.zero(("x",)),
        layouts=(Layout({}),),
    )
    space = mx.center
    assert isinstance(space, TensorProductSpace | type(space))
    assert dec.storage_shape(space) == (8,)
    data = jnp.arange(8.0)
    assert jnp.array_equal(dec.unpad(dec.pad(data, space), space), data)
