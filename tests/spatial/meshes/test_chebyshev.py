"""Tests for ChebyshevMesh (spatial/meshes/chebyshev.py)."""
from fractions import Fraction

import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.traits import HaloStrategy
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.coefficient import ChebyshevSpace
from fridom.spatial.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def mesh():
    return ChebyshevMesh(N, (0, 1), name="z")


def test_bounded_by_construction(mesh):
    assert mesh.periodic is False
    assert mesh.n_cells == N
    assert mesh.dim == 1


# ================================================================
#  Lobatto / outer family
# ================================================================
def test_lobatto_is_outer(mesh):
    assert mesh.lobatto is mesh.outer
    assert mesh.lobatto.shape == (N + 1,)


def test_bc_structured_lobatto(mesh):
    dirichlet = mesh.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    assert dirichlet.shape == (N - 1,)


# ================================================================
#  Restricted space family (owner decision)
# ================================================================
@pytest.mark.parametrize("factory", ["center", "left", "right", "inner"])
def test_no_cell_nodal_family(mesh, factory):
    with pytest.raises(ValueError, match="restricted"):
        getattr(mesh, factory)


@pytest.mark.parametrize("factory", ["cell_avg", "face_avg"])
def test_no_average_family(mesh, factory):
    with pytest.raises(ValueError, match="restricted"):
        getattr(mesh, factory)


def test_fourier_raises(mesh):
    with pytest.raises(ValueError, match="periodic"):
        mesh.fourier(origin=mesh.lobatto)


def test_sine_is_admissible(mesh):
    origin = mesh.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    assert mesh.sine(origin=origin).shape == (N - 1,)


# ================================================================
#  Chebyshev coefficient factory
# ================================================================
def test_chebyshev_space(mesh):
    space = mesh.chebyshev(origin=mesh.lobatto)
    assert type(space) is ChebyshevSpace
    assert space.shape == (N + 1,)
    assert space is mesh.chebyshev(origin=mesh.lobatto)
    assert space.origin is mesh.lobatto


def test_chebyshev_origin_must_be_outer_family(mesh):
    with pytest.raises(TypeError, match="nodal or average"):
        mesh.chebyshev(origin=mesh.constant)


def test_chebyshev_origin_must_live_here(mesh):
    other = ChebyshevMesh(N, (0, 1), name="z")
    with pytest.raises(ValueError, match="lives on"):
        mesh.chebyshev(origin=other.lobatto)


def test_chebyshev_rejects_non_lobatto_origin(mesh):
    other = IntervalMesh(N, (0, 1), periodic=False, name="z")
    with pytest.raises(ValueError, match="lives on"):
        mesh.chebyshev(origin=other.center)


# ================================================================
#  Refinement, traits, repr
# ================================================================
def test_refined_is_designed_for(mesh):
    with pytest.raises(NotImplementedError, match="designed-for"):
        mesh.refined(Fraction(3, 2))


def test_traits_are_transpose_first(mesh):
    for space in (mesh.lobatto, mesh.chebyshev(origin=mesh.lobatto)):
        traits = mesh.decomposition_traits(space)
        assert traits.strategies == (
            HaloStrategy.TRANSPOSE, HaloStrategy.LOCAL)


def test_traits_constant_is_local_only(mesh):
    traits = mesh.decomposition_traits(mesh.constant)
    assert traits.strategies == (HaloStrategy.LOCAL,)


def test_repr(mesh):
    assert repr(mesh) == "ChebyshevMesh(z: n=8, extent=(0, 1), bounded)"
