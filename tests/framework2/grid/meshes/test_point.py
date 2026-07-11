"""Tests for PointMesh (framework2/grid/meshes/point.py)."""
import pytest

from fridom.spatial.decomposition.traits import HaloStrategy
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.point import PointMesh
from fridom.spatial.spaces.nodal import NodeSet, PointValues


@pytest.fixture
def two_points():
    return PointMesh(((0.0,), (1.0,)), name="x")


# ================================================================
#  Construction and descriptors
# ================================================================
def test_descriptors(two_points):
    assert two_points.dim == 0
    assert two_points.n_points == 2
    assert two_points.positions == ((0.0,), (1.0,))
    assert two_points.names == ("x",)


def test_positions_are_coerced_to_floats():
    mesh = PointMesh(((0, 1), (2, 3)), name="p")
    assert mesh.positions == ((0.0, 1.0), (2.0, 3.0))


def test_empty_point_mesh_is_fine():
    assert PointMesh((), name="x").n_points == 0


def test_inconsistent_position_lengths_rejected():
    with pytest.raises(ValueError, match="same number"):
        PointMesh(((0.0,), (1.0, 2.0)), name="x")


# ================================================================
#  Boundary (a 0D mesh has no boundary)
# ================================================================
def test_boundary_is_the_empty_point_mesh(two_points):
    boundary = two_points.boundary
    assert isinstance(boundary, PointMesh)
    assert boundary.n_points == 0
    assert boundary is two_points.boundary


# ================================================================
#  The points space
# ================================================================
def test_points_is_interned(two_points):
    assert two_points.points is two_points.points
    assert isinstance(two_points.points, PointValues)
    assert two_points.points.mesh is two_points


def test_points_shape_and_node_set(two_points):
    assert two_points.points.shape == (2,)
    assert two_points.points.node_set is NodeSet.POINTS


def test_empty_points_space_has_zero_dofs():
    mesh = IntervalMesh(8, (0, 1), name="x")
    assert mesh.boundary.points.shape == (0,)


def test_points_scalar_variants(two_points):
    assert two_points.points.as_complex() is two_points.points.as_complex()
    assert two_points.points.as_complex().as_real() is two_points.points


# ================================================================
#  Traits and repr
# ================================================================
def test_traits_are_local_only(two_points):
    traits = two_points.decomposition_traits(two_points.points)
    assert traits.strategies == (HaloStrategy.LOCAL,)
    traits = two_points.decomposition_traits(two_points.constant)
    assert traits.strategies == (HaloStrategy.LOCAL,)


def test_traits_reject_foreign_spaces(two_points):
    other = PointMesh(((0.0,),), name="q")
    with pytest.raises(ValueError, match="lives on"):
        two_points.decomposition_traits(other.points)


def test_repr(two_points):
    assert repr(two_points) == "PointMesh(x: 2 points)"
