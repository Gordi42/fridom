"""Tests for fridom.framework2.grid.operators.interp."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def interp():
    return LinearInterp()


# ================================================================
#  Construction and static surface
# ================================================================
def test_target_validation():
    assert LinearInterp().target is None
    assert LinearInterp(target=NodeSet.OUTER).target is NodeSet.OUTER
    with pytest.raises(TypeError, match="NodeSet"):
        LinearInterp(target="outer")


def test_dispatch_kind_and_requirements(interp, mx):
    assert interp.dispatch_kind == "interpolate"
    assert interp.requirements(mx.center).halo == 1
    assert interp.requirements(mx.center).layout == "any"


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(interp, mx):
    assert interp.codomain(mx.center) is mx.right
    assert interp.codomain(mx.right) is mx.center


def test_codomain_bounded(interp, my):
    assert interp.codomain(my.center) is my.inner
    assert interp.codomain(my.outer) is my.center
    assert interp.codomain(my.inner) is my.center


def test_codomain_preserves_scalars(interp, mx):
    assert interp.codomain(mx.right.as_complex()) is (
        mx.center.as_complex())


def test_codomain_outer_variant(my):
    outer = LinearInterp(target=NodeSet.OUTER)
    assert outer.codomain(my.center) is my.outer


def test_codomain_outer_variant_rejects_other_domains(mx, my):
    outer = LinearInterp(target=NodeSet.OUTER)
    with pytest.raises(SpaceMismatchError, match="Center -> Outer"):
        outer.codomain(mx.center)  # periodic has no Outer
    with pytest.raises(SpaceMismatchError, match="Center -> Outer"):
        outer.codomain(my.inner)


def test_codomain_rejects_average_and_unlisted(interp, mx, my):
    with pytest.raises(SpaceMismatchError, match="reconstruct"):
        interp.codomain(mx.cell_avg)
    with pytest.raises(SpaceMismatchError,
                       match="no interpolate signature"):
        interp.codomain(my.left)


# ================================================================
#  Application
# ================================================================
def test_periodic_center_to_right_wraps(interp, mx):
    grid = Grid((mx,))
    f = grid.create_field(data=jnp.arange(8.0))
    g = interp["x"](f)
    assert g.function_space.bare is mx.right
    expected = 0.5 * (jnp.arange(8.0)
                      + jnp.roll(jnp.arange(8.0), -1))
    assert jnp.allclose(g.data, expected)


def test_bounded_center_to_inner(interp, my):
    grid = Grid((my,))
    f = grid.create_field(data=jnp.arange(8.0))
    g = interp["y"](f)
    assert g.function_space.bare is my.inner
    assert jnp.allclose(g.data, jnp.arange(7) + 0.5)


def test_bounded_outer_variant_extrapolates_boundary_faces(my):
    grid = Grid((my,))
    outer = LinearInterp(target=NodeSet.OUTER)
    f = grid.create_field(init=lambda y: 2.0 * y + 1.0)
    g = outer["y"](f)
    assert g.function_space.bare is my.outer
    # exact for linear data, including the extrapolated boundaries
    y_outer = grid.evaluation_nodes(my.outer).data
    assert jnp.allclose(g.data, 2.0 * y_outer + 1.0)


def test_metadata_is_preserved(interp, mx):
    grid = Grid((mx,))
    f = grid.create_field(name="u", units="m/s")
    g = interp["x"](f)
    assert g.metadata == f.metadata  # same-quantity rule
