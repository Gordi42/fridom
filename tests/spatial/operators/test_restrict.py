"""Tests for fridom.spatial.operators.restrict."""
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.base import OperatorRequirements
from fridom.spatial.operators.restrict import Restriction
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import NodeSet

N = 6


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def mz():
    return IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")


@pytest.fixture
def walled(mz):
    return Grid((mz,))


@pytest.fixture
def restrict():
    return Restriction()


# ================================================================
#  Identity, interning, requirements
# ================================================================
def test_is_interned_and_carries_the_restrict_kind():
    assert Restriction() is Restriction()
    assert Restriction().dispatch_kind == "restrict"


def test_requirements_are_halo_zero(restrict, mz):
    # a pure interior-node selection reads no ghost layers
    req = restrict.requirements(mz.outer)
    assert isinstance(req, OperatorRequirements)
    assert req.halo == 0
    assert req.layout == "any"


# ================================================================
#  Signature: Outer -> Inner only
# ================================================================
def test_codomain_maps_outer_to_inner(restrict, mz):
    assert restrict.codomain(mz.outer) is mz.inner


def test_codomain_preserves_complex(restrict, mz):
    out = restrict.codomain(mz.outer.as_complex())
    assert out is mz.inner.as_complex()
    assert out.scalars is Scalars.COMPLEX


@pytest.mark.parametrize("factory", ["center", "left", "right", "inner"])
def test_codomain_rejects_non_outer_node_sets(restrict, mz, factory):
    with pytest.raises(SpaceMismatchError, match="Outer"):
        restrict.codomain(getattr(mz, factory))


def test_codomain_rejects_a_periodic_factor(restrict, mx):
    # a periodic mesh carries no Outer face set at all
    with pytest.raises(SpaceMismatchError, match="Outer"):
        restrict.codomain(mx.center)


def test_codomain_rejects_an_outer_without_an_inner_sibling(restrict):
    # a ChebyshevMesh carries the Lobatto Outer but no interior-face
    # Inner family: the row un-seeds itself with a taught message
    mesh = ChebyshevMesh(6, (0.0, 1.0), name="z")
    with pytest.raises(SpaceMismatchError, match="no Inner"):
        restrict.codomain(mesh.outer)


def test_codomain_rejects_a_dirichlet_outer(restrict, mz):
    # a Dirichlet condition drops the member boundary DOF of Outer, so
    # the interior-selection alignment no longer holds
    with pytest.raises(SpaceMismatchError, match="boundary"):
        restrict.codomain(mz.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))


# ================================================================
#  Exactness: the restriction drops the two boundary faces
# ================================================================
def test_restriction_selects_the_interior_faces_exactly(walled, mz):
    w = walled.create_field(mz.outer, init=lambda z: z**2 - 0.3 * z)
    r = Restriction()["z"](w)
    assert r.function_space.bare.factor("z") is mz.inner
    outer = np.asarray(w.data)
    inner = np.asarray(r.data)
    # Inner[m] == Outer[m + 1]: the n + 1 faces minus the two walls
    assert np.allclose(inner, outer[1:-1])
    assert inner.shape[0] == outer.shape[0] - 2


def test_restriction_is_exact_on_a_stretched_mesh():
    # a pure node selection carries no metric, so it is exact on a
    # stretched (mapped) mesh as well as a uniform one
    mesh = MappedIntervalMesh(
        N, (0.0, 1.0), lambda s: s + 0.1 * np.sin(2 * np.pi * s),
        periodic=False, name="z")
    grid = Grid((mesh,))
    w = grid.create_field(mesh.outer, init=lambda z: 2.0 * z + 1.0)
    r = Restriction()["z"](w)
    assert np.allclose(np.asarray(r.data),
                       np.asarray(w.data)[1:-1])
