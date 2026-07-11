"""Tests for the average spaces (spatial/spaces/average.py)."""
import pytest

from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.spatial.spaces.nodal import NodalSpace

N = 8


@pytest.fixture
def periodic():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 1), periodic=False, name="x")


# ================================================================
#  Shape table (section 3.5)
# ================================================================
def test_cell_avg_shape(periodic, bounded):
    assert periodic.cell_avg.shape == (N,)
    assert bounded.cell_avg.shape == (N,)


def test_face_avg_shape(periodic, bounded):
    assert periodic.face_avg.shape == (N,)
    assert bounded.face_avg.shape == (N - 1,)


# ================================================================
#  Interning and identity
# ================================================================
def test_interning(periodic):
    assert periodic.cell_avg is periodic.cell_avg
    assert periodic.face_avg is periodic.face_avg
    assert periodic.cell_avg is not periodic.face_avg


def test_average_is_not_nodal(periodic):
    # distinct from nodal even at equal DOF counts (averages have
    # no position)
    assert isinstance(periodic.cell_avg, AverageSpace)
    assert not isinstance(periodic.cell_avg, NodalSpace)
    assert periodic.cell_avg is not periodic.center
    assert periodic.face_avg is not periodic.right


def test_classes(periodic):
    assert type(periodic.cell_avg) is CellAvg
    assert type(periodic.face_avg) is FaceAvg


# ================================================================
#  Defining attributes
# ================================================================
def test_defaults(periodic):
    assert periodic.cell_avg.scalars is Scalars.REAL
    assert periodic.cell_avg.bc.is_free
    assert periodic.cell_avg.mesh is periodic


def test_scalar_variants(periodic):
    complex_avg = periodic.cell_avg.as_complex()
    assert complex_avg is periodic.cell_avg.as_complex()
    assert complex_avg.as_real() is periodic.cell_avg
    assert complex_avg.shape == periodic.cell_avg.shape


def test_repr(periodic):
    assert repr(periodic.cell_avg) == "CellAvg(x)"
    assert repr(periodic.face_avg) == "FaceAvg(x)"
