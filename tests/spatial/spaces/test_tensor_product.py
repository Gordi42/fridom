"""Tests for the product space and join (spaces/tensor_product.py)."""
import gc
import weakref

import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import _bindable_names
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.function_space import FunctionSpace
from fridom.spatial.spaces.tensor_product import (
    SpaceLike,
    TensorProductSpace,
    join,
    join_factor,
    lifts_to,
    require_same_layout,
)

N = 8


@pytest.fixture
def mx():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def my():
    return IntervalMesh(N + 8, (0, 2), name="y")


@pytest.fixture
def mz():
    return ChebyshevMesh(N, (0, 1), name="z")


# ================================================================
#  Construction: flat, associative, interned
# ================================================================
def test_mul_is_flat_and_associative(mx, my, mz):
    left = (mx.center * my.center) * mz.lobatto
    right = mx.center * (my.center * mz.lobatto)
    assert left is right
    assert left.factors == (mx.center, my.center, mz.lobatto)


def test_of_and_mul_agree(mx, my):
    assert TensorProductSpace.of(mx.center, my.center) is (
        mx.center * my.center)


def test_no_factor_is_a_product(mx, my, mz):
    product = mx.center * my.center * mz.lobatto
    assert all(isinstance(factor, FunctionSpace)
               for factor in product.factors)


def test_single_factor_returns_the_factor_itself(mx):
    assert TensorProductSpace.of(mx.center) is mx.center


def test_of_a_product_returns_it(mx, my):
    product = mx.center * my.center
    assert TensorProductSpace.of(product) is product


def test_empty_product_rejected():
    with pytest.raises(ValueError, match="empty product"):
        TensorProductSpace.of()


def test_non_space_rejected(mx):
    with pytest.raises(TypeError, match="function spaces"):
        TensorProductSpace.of(mx.center, "y")


def test_duplicate_names_rejected(mx):
    # one factor per mesh: names are per mesh
    with pytest.raises(ValueError, match="duplicate"):
        _ = mx.center * mx.right
    other = IntervalMesh(N, (0, 5), name="x")
    with pytest.raises(ValueError, match="duplicate"):
        _ = mx.center * other.center


def test_weak_interning(mx, my):
    product = mx.center * my.center
    ref = weakref.ref(product)
    assert ref() is product
    del product
    gc.collect()
    assert ref() is None


# ================================================================
#  Properties
# ================================================================
def test_factors_names_shape_ndim(mx, my, mz):
    product = mx.center * my.center * mz.lobatto
    assert product.factors == (mx.center, my.center, mz.lobatto)
    assert product.names == ("x", "y", "z")
    assert product.shape == (N, N + 8, N + 1)
    assert product.ndim == 3
    assert len(product) == 3
    assert tuple(product) == product.factors


def test_scalars_is_derived(mx, my):
    real = mx.center * my.center
    assert real.scalars is Scalars.REAL
    mixed = mx.center.as_complex() * my.center
    assert mixed.scalars is Scalars.COMPLEX


def test_factor_by_name(mx, my):
    product = mx.center * my.center
    assert product.factor("x") is mx.center
    assert product.factor("y") is my.center
    with pytest.raises(KeyError, match="no factor"):
        product.factor("t")


# ================================================================
#  Constancy predicates
# ================================================================
def test_active_axis_names_pure_nodal(mx, my, mz):
    product = mx.center * my.center * mz.lobatto
    assert product.active_axis_names == ("x", "y", "z")


def test_active_axis_names_drops_constant(mx, my):
    product = mx.center * my.constant
    assert product.active_axis_names == ("x",)


def test_has_constant_factor_mixed_vs_pure(mx, my):
    assert (mx.center * my.center).has_constant_factor is False
    assert (mx.center * my.constant).has_constant_factor is True


def test_active_axis_names_matches_bindable_names(mx, my):
    pure = mx.center * my.center
    mixed = mx.center * my.constant
    assert pure.active_axis_names == _bindable_names(pure)
    assert mixed.active_axis_names == _bindable_names(mixed)


# ================================================================
#  replace and as_complex
# ================================================================
def test_replace_builds_the_interned_codomain(mx, my):
    product = mx.center * my.center
    replaced = product.replace(x=mx.right)
    assert replaced is mx.right * my.center
    assert replaced.factor("y") is my.center


def test_replace_with_constant(mx, my):
    # the integrate codomain
    product = mx.center * my.center
    assert product.replace(x=mx.constant) is mx.constant * my.center


def test_replace_requires_the_same_mesh(mx, my):
    product = mx.center * my.center
    other = IntervalMesh(N, (0, 1), name="x")
    with pytest.raises(ValueError, match="same mesh"):
        product.replace(x=other.center)


def test_replace_unknown_name(mx, my):
    with pytest.raises(KeyError, match="no factor"):
        (mx.center * my.center).replace(t=mx.right)


def test_replace_non_space(mx, my):
    with pytest.raises(TypeError, match="factor space"):
        (mx.center * my.center).replace(x="right")


def test_as_complex(mx, my):
    product = mx.center * my.center
    complexified = product.as_complex()
    assert complexified is (
        mx.center.as_complex() * my.center.as_complex())
    assert complexified.scalars is Scalars.COMPLEX


# ================================================================
#  Layout protocol (section 5.1)
# ================================================================
def test_bare_products_have_no_layout(mx, my):
    product = mx.center * my.center
    assert product.layout is None
    assert product.bare is product


def test_with_layout_is_interned(mx, my):
    product = mx.center * my.center
    laid_out = product.with_layout("L0")
    assert laid_out is product.with_layout("L0")
    assert laid_out is not product
    assert laid_out.layout == "L0"
    assert laid_out.bare is product
    assert laid_out.with_layout(None) is product
    assert laid_out.shape == product.shape


def test_factor_returns_bare_factors(mx, my):
    laid_out = (mx.center * my.center).with_layout("L0")
    assert laid_out.factor("x") is mx.center
    assert laid_out.factors == (mx.center, my.center)


def test_replace_and_as_complex_preserve_the_layout(mx, my):
    laid_out = (mx.center * my.center).with_layout("L0")
    assert laid_out.replace(x=mx.right).layout == "L0"
    assert laid_out.as_complex().layout == "L0"


def test_of_rejects_laid_out_inputs(mx, my):
    with pytest.raises(ValueError, match="bare"):
        TensorProductSpace.of(mx.center.with_layout("L0"), my.center)
    laid_out = (mx.center * my.center).with_layout("L0")
    mz = IntervalMesh(N, (0, 1), name="z")
    with pytest.raises(ValueError, match="bare"):
        TensorProductSpace.of(laid_out, mz.center)


# ================================================================
#  Repr and SpaceLike
# ================================================================
def test_repr(mx, my):
    assert repr(mx.center * my.center) == "Center(x) ⊗ Center(y)"


def test_repr_with_layout(mx, my):
    laid_out = (mx.center * my.center).with_layout("L0")
    assert repr(laid_out) == "Center(x) ⊗ Center(y) [layout='L0']"


def test_space_like_alias(mx, my):
    assert isinstance(mx.center, SpaceLike)
    assert isinstance(mx.center * my.center, SpaceLike)


# ================================================================
#  The join: per-factor least upper bound
# ================================================================
def test_join_identical_spaces(mx, my):
    product = mx.center * my.center
    assert join(product, product) is product
    assert join(mx.center, mx.center) is mx.center


def test_join_constant_broadcast(mx, my):
    full = mx.center * my.center
    partial = mx.center * my.constant
    assert join(full, partial) is full
    assert join(partial, full) is full


def test_join_complex_promotion(mx):
    assert join(mx.center, mx.center.as_complex()) is (
        mx.center.as_complex())


def test_join_chained_lifts(mx, my):
    # constant + complex promotion combine per factor
    a = mx.center * my.constant.as_complex()
    b = mx.center * my.center
    assert join(a, b) is mx.center * my.center.as_complex()


def test_join_preserves_the_common_layout(mx, my):
    a = (mx.center * my.constant).with_layout("L0")
    b = (mx.center * my.center).with_layout("L0")
    assert join(a, b) is b


def test_join_layout_mismatch_raises(mx, my):
    product = mx.center * my.center
    with pytest.raises(SpaceMismatchError, match="reshard"):
        join(product, product.with_layout("L0"))
    with pytest.raises(SpaceMismatchError, match="same space"):
        join(product.with_layout("L0"), product.with_layout("L1"))


def test_join_mismatch_raises_with_a_factor_diff(mx, my):
    a = mx.right * my.center
    b = mx.center * my.center
    with pytest.raises(SpaceMismatchError,
                       match=r"x: Right\(x\) vs Center\(x\)") as info:
        join(a, b, operation="+")
    error = info.value
    assert error.left is a
    assert error.right is b
    assert error.operation == "+"
    assert error.mismatched_names == ("x",)
    assert "y agree" in str(error)
    assert ".to(" in str(error)


def test_join_same_names_different_meshes_raises(mx, my):
    # equal name sets, distinct mesh objects: still a mesh mismatch
    other = IntervalMesh(N + 8, (0, 2), name="y")
    a = mx.center * my.center
    b = mx.center * other.center
    with pytest.raises(SpaceMismatchError, match="meshes differ") as e:
        join(a, b)
    assert e.value.mismatched_names == ("x", "y")


def test_join_mesh_mismatch_raises(mx, my, mz):
    a = mx.center * my.center
    b = mx.center * mz.lobatto
    with pytest.raises(SpaceMismatchError, match="meshes differ"):
        join(a, b)
    with pytest.raises(SpaceMismatchError, match="meshes differ"):
        join(mx.center, a)


def test_join_average_vs_nodal_is_not_lifted(mx):
    with pytest.raises(SpaceMismatchError, match=r"use \.to"):
        join(mx.center, mx.cell_avg)


# ================================================================
#  The per-factor lifts
# ================================================================
def test_join_factor_lub(mx):
    assert join_factor(mx.center, mx.center) is mx.center
    assert join_factor(mx.constant, mx.center) is mx.center
    assert join_factor(mx.center, mx.constant) is mx.center
    assert join_factor(mx.constant, mx.constant) is mx.constant
    assert join_factor(
        mx.constant.as_complex(), mx.center) is mx.center.as_complex()
    assert join_factor(mx.center, mx.right) is None


def test_join_factor_cross_mesh_is_none(mx, my):
    assert join_factor(mx.center, my.center) is None


def test_lifts_to(mx):
    assert lifts_to(mx.center, mx.center)
    assert lifts_to(mx.constant, mx.center)
    assert lifts_to(mx.center, mx.center.as_complex())
    assert not lifts_to(mx.center, mx.constant)
    assert not lifts_to(mx.center.as_complex(), mx.center)
    assert not lifts_to(mx.center, mx.right)


def test_require_same_layout(mx):
    require_same_layout(mx.center, mx.center)
    with pytest.raises(SpaceMismatchError, match="layouts differ"):
        require_same_layout(mx.center, mx.center.with_layout("L0"),
                            operation="+")
