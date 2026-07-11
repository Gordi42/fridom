"""Tests for FunctionSpace (spatial/spaces/function_space.py)."""
import pytest

from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import _bindable_names
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import Center, NodeSet
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

N = 8


@pytest.fixture
def mesh():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 1), periodic=False, name="x")


# ================================================================
#  Guarded construction
# ================================================================
def test_direct_construction_raises(mesh):
    bc = BCStructure(())
    with pytest.raises(TypeError, match="factory"):
        Center(mesh, Scalars.REAL, bc)
    with pytest.raises(TypeError, match="factory"):
        Center(mesh, Scalars.REAL, bc, _token=object())


# ================================================================
#  Identity semantics
# ================================================================
def test_equality_is_identity(mesh):
    space = mesh.center
    assert space == space  # noqa: PLR0124 — identity semantics
    assert space is mesh.center
    other = IntervalMesh(N, (0, 1), name="x")
    assert space != other.center
    assert hash(space) != hash(other.center)
    assert (space == "Center(x)") is False


def test_eq_is_not_the_object_default(mesh):
    # fridom's structural-equality walk keys off this
    assert type(mesh.center).__eq__ is not object.__eq__
    assert type(mesh.center).__hash__ is not object.__hash__


# ================================================================
#  Defining attributes
# ================================================================
def test_mesh_backreference(mesh):
    assert mesh.center.mesh is mesh


def test_scalars_and_bc_defaults(mesh):
    assert mesh.center.scalars is Scalars.REAL
    assert mesh.center.bc.is_free


def test_variance_is_none(mesh):
    assert mesh.center.variance is None


# ================================================================
#  Product protocol (single factor == degenerate product)
# ================================================================
def test_factors_is_self(mesh):
    assert mesh.center.factors == (mesh.center,)


def test_names_are_the_mesh_names(mesh):
    assert mesh.center.names == ("x",)


def test_factor_by_name(mesh):
    assert mesh.center.factor("x") is mesh.center
    with pytest.raises(KeyError, match="no factor"):
        mesh.center.factor("y")


def test_mul_returns_a_product(mesh):
    other = IntervalMesh(N, (0, 1), name="y")
    product = mesh.center * other.center
    assert isinstance(product, TensorProductSpace)
    assert product.factors == (mesh.center, other.center)


# ================================================================
#  Constancy predicates
# ================================================================
def test_is_constant_false_for_full_factor(mesh):
    assert mesh.center.is_constant is False


def test_active_axis_names_single_factor(mesh):
    assert mesh.center.active_axis_names == ("x",)
    assert mesh.constant.active_axis_names == ()


def test_has_constant_factor_bare_space(mesh):
    assert mesh.center.has_constant_factor is False
    assert mesh.constant.has_constant_factor is True


def test_active_axis_names_matches_bindable_names(mesh):
    assert mesh.center.active_axis_names == _bindable_names(mesh.center)
    assert mesh.constant.active_axis_names == _bindable_names(
        mesh.constant)


# ================================================================
#  Scalar variants
# ================================================================
def test_as_complex_is_interned(mesh):
    complex_center = mesh.center.as_complex()
    assert complex_center is mesh.center.as_complex()
    assert complex_center is not mesh.center
    assert complex_center.scalars is Scalars.COMPLEX


def test_as_complex_is_idempotent(mesh):
    complex_center = mesh.center.as_complex()
    assert complex_center.as_complex() is complex_center
    assert mesh.center.as_real() is mesh.center


def test_scalar_roundtrip(mesh):
    assert mesh.center.as_complex().as_real() is mesh.center


def test_bc_survives_scalar_variants(bounded):
    dirichlet = bounded.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    assert dirichlet.as_complex().bc == dirichlet.bc
    assert dirichlet.as_complex().shape == dirichlet.shape
    assert dirichlet.as_complex() is not bounded.outer.as_complex()


# ================================================================
#  Layout protocol
# ================================================================
def test_bare_space_has_no_layout(mesh):
    assert mesh.center.layout is None
    assert mesh.center.bare is mesh.center


def test_with_layout_is_interned(mesh):
    laid_out = mesh.center.with_layout(("x", "p0"))
    assert laid_out is mesh.center.with_layout(("x", "p0"))
    assert laid_out is not mesh.center
    assert laid_out.layout == ("x", "p0")
    assert laid_out.bare is mesh.center
    assert laid_out.with_layout(None) is mesh.center


def test_value_equal_layouts_intern_together(mesh):
    # Layout is an opaque hashable value; equal values, same space
    a = mesh.center.with_layout(("x", "p" + "0"))
    b = mesh.center.with_layout(("x", "p0"))
    assert a is b


def test_distinct_layouts_are_distinct_spaces(mesh):
    a = mesh.center.with_layout("L0")
    b = mesh.center.with_layout("L1")
    assert a is not b


def test_laid_out_factors_are_bare(mesh):
    laid_out = mesh.center.with_layout("L0")
    assert laid_out.factors == (mesh.center,)
    assert laid_out.factor("x") is mesh.center


def test_layout_preserved_by_scalar_variants(mesh):
    laid_out = mesh.center.with_layout("L0")
    assert laid_out.as_complex().layout == "L0"
    assert laid_out.as_complex() is mesh.center.as_complex(
        ).with_layout("L0")


def test_shape_is_layout_independent(mesh):
    assert mesh.center.with_layout("L0").shape == mesh.center.shape


# ================================================================
#  Repr
# ================================================================
def test_repr_spot_checks(mesh, bounded):
    assert repr(mesh.center) == "Center(x)"
    assert repr(mesh.center.as_complex()) == "Center(x, complex)"
    assert repr(bounded.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)) == (
        "Outer(x, bc=(DIRICHLET, DIRICHLET))")
    assert repr(mesh.center.with_layout("L0")) == (
        "Center(x, layout='L0')")
