"""Tests for ConstantSpace (spatial/spaces/constant.py)."""
import pytest

from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.point import PointMesh
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace

N = 8


@pytest.fixture
def mesh():
    return IntervalMesh(N, (0, 1), name="x")


def test_shape_is_one(mesh):
    assert mesh.constant.shape == (1,)


def test_defaults(mesh):
    assert mesh.constant.scalars is Scalars.REAL
    assert mesh.constant.bc.is_free
    assert mesh.constant.mesh is mesh


def test_one_per_mesh_and_scalars(mesh):
    assert mesh.constant is mesh.constant
    complex_constant = mesh.constant.as_complex()
    assert complex_constant is mesh.constant.as_complex()
    assert complex_constant is not mesh.constant
    assert complex_constant.as_real() is mesh.constant
    assert complex_constant.shape == (1,)


def test_universal_across_mesh_types():
    # 'constant along this factor' makes sense for every mesh
    for mesh in (
        IntervalMesh(N, (0, 1), name="x"),
        ChebyshevMesh(N, (0, 1), name="z"),
        PointMesh(((0.0,),), name="p"),
    ):
        assert isinstance(mesh.constant, ConstantSpace)
        assert mesh.constant is mesh.constant


def test_not_a_nodal_space(mesh):
    # distinct from PointValues: a geometry-less bulk reduction
    assert not isinstance(mesh.constant, NodalSpace)


def test_is_constant_true(mesh):
    # the public predicate replacing isinstance(x, ConstantSpace)
    assert mesh.constant.is_constant is True
    assert mesh.center.is_constant is False


def test_repr(mesh):
    assert repr(mesh.constant) == "Constant(x)"
