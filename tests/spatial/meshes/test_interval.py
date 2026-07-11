"""Tests for IntervalMesh (spatial/meshes/interval.py)."""
from fractions import Fraction

import pytest

from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.point import PointMesh

N = 8


@pytest.fixture
def periodic():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 2), periodic=False, name="x")


def test_dx(periodic, bounded):
    assert periodic.dx == pytest.approx(1 / N)
    assert bounded.dx == pytest.approx(2 / N)


def test_periodic_default_true():
    assert IntervalMesh(N, (0, 1), name="x").periodic is True


# ================================================================
#  Boundary
# ================================================================
def test_boundary_is_stable(bounded):
    assert bounded.boundary is bounded.boundary


def test_bounded_boundary_has_the_two_endpoints(bounded):
    boundary = bounded.boundary
    assert isinstance(boundary, PointMesh)
    assert boundary.dim == 0
    assert boundary.n_points == 2
    assert boundary.positions == ((0.0,), (2.0,))
    assert boundary.names == ("x",)


def test_periodic_boundary_is_empty(periodic):
    assert periodic.boundary.n_points == 0


def test_boundary_prints_as_boundary_of_parent(bounded):
    assert repr(bounded.boundary) == "PointMesh(boundary(x): 2 points)"
    assert repr(bounded.boundary.points) == "PointValues(boundary(x))"


# ================================================================
#  Refinement (iteration 1 on IntervalMesh)
# ================================================================
def test_refined_scales_the_cell_count(periodic):
    finer = periodic.refined(Fraction(3, 2))
    assert isinstance(finer, IntervalMesh)
    assert finer.n_cells == 12
    assert finer.extent == periodic.extent
    assert finer.periodic == periodic.periodic
    assert finer.names == periodic.names


def test_refined_accepts_ints(periodic):
    assert periodic.refined(2).n_cells == 2 * N


def test_refined_is_memoized_per_factor(periodic):
    assert periodic.refined(Fraction(3, 2)) is periodic.refined(
        Fraction(3, 2))
    assert periodic.refined(2) is periodic.refined(Fraction(2))
    assert periodic.refined(2) is not periodic.refined(Fraction(3, 2))


def test_refined_is_a_distinct_first_class_mesh(periodic):
    finer = periodic.refined(2)
    assert finer is not periodic
    assert finer != periodic
    # its spaces are interned on it and distinct from the parent's
    assert finer.center is finer.center
    assert finer.center is not periodic.center
    assert finer.center.shape == (2 * N,)


def test_refined_from(periodic):
    assert periodic.refined_from is None
    assert periodic.refined(2).refined_from is periodic


def test_refined_must_be_integral(periodic):
    with pytest.raises(ValueError, match="integral"):
        periodic.refined(Fraction(1, 3))


def test_refined_must_be_positive(periodic):
    with pytest.raises(ValueError, match="positive"):
        periodic.refined(Fraction(-1, 2))


def test_refined_rejects_inexact_factors(periodic):
    with pytest.raises(TypeError, match="exact Fractions"):
        periodic.refined(1.5)
    with pytest.raises(TypeError, match="exact Fractions"):
        periodic.refined(True)  # noqa: FBT003 — bool rejection
