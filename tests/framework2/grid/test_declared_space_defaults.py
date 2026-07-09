"""Tests for the default ``("declared_space", mesh)`` resolver rows.

A bare ``Grid`` now seeds one resolver row per structured mesh
factor (grid.py, ``_default_registry``), so the model-layer space
patterns (``Collocated()`` / ``Staggered(...)`` / ``Profile(...)``)
resolve out of the box: COLLOCATED -> the center/nodal family
(ChebyshevMesh: the outer/Lobatto family), STAGGERED -> the face
family — Right on periodic meshes, Inner on bounded ones (a C-grid
wall-normal velocity carries interior faces only; the wall value is
a boundary condition, not a DOF) — and an error on ChebyshevMesh
(no face spaces).
"""
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.model.space_patterns import (
    Collocated,
    Profile,
    Staggered,
)


# ================================================================
#  Fixtures: bare grids, NO manual resolver seeding
# ================================================================
@pytest.fixture(scope="module")
def meshes():
    x = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    z = IntervalMesh(8, (-1.0, 0.0), periodic=False, name="z")
    return x, z


@pytest.fixture(scope="module")
def grid(meshes):
    return Grid(meshes)


@pytest.fixture(scope="module")
def cheb_grid():
    return Grid((ChebyshevMesh(8, (0.0, 1.0), name="s"),))


# ================================================================
#  The seeded rows
# ================================================================
def test_resolver_rows_are_seeded_per_mesh(grid, meshes):
    for mesh in meshes:
        assert ("declared_space", mesh) in grid.dispatch


# ================================================================
#  Interval meshes: center / right families
# ================================================================
def test_collocated_resolves_to_the_center_family(grid, meshes):
    x, z = meshes
    space = Collocated().resolve(grid)
    assert space.factors == (x.center, z.center)


def test_staggered_resolves_right_periodic_inner_bounded(grid,
                                                         meshes):
    # periodic axes stagger to Right; bounded axes to Inner (the
    # wall-normal velocity's wall values are BCs, not DOFs)
    x, z = meshes
    assert Staggered("x").resolve(grid).factors == (x.right,
                                                    z.center)
    assert Staggered("z").resolve(grid).factors == (x.center,
                                                    z.inner)
    assert Staggered("x", "z").resolve(grid).factors == (x.right,
                                                         z.inner)


def test_bounded_staggered_bc_resolves_on_inner(grid, meshes):
    from fridom.framework2.grid.spaces.nodal import (  # noqa: PLC0415
        NodeSet,
    )
    x, z = meshes
    space = Staggered("z", bc={"z": BC.DIRICHLET}).resolve(grid)
    assert space.factors == (
        x.center, z.nodal(NodeSet.INNER, bc=BC.DIRICHLET))


def test_profile_mixes_constant_and_center(grid, meshes):
    x, z = meshes
    assert Profile("z").resolve(grid).factors == (x.constant,
                                                  z.center)
    assert Profile().resolve(grid).factors == (x.constant,
                                               z.constant)


def test_unmatched_names_degrade_gracefully(grid, meshes):
    x, z = meshes
    # a name absent from the grid is simply unmatched (D1.2)
    assert Staggered("y").resolve(grid).factors == (x.center,
                                                    z.center)


def test_bc_entries_reach_the_resolver(grid, meshes):
    from fridom.framework2.grid.spaces.nodal import (  # noqa: PLC0415
        NodeSet,
    )
    x, z = meshes
    space = Collocated(bc={"z": BC.DIRICHLET}).resolve(grid)
    assert space.factors == (
        x.center, z.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))


def test_repeated_resolution_is_interned(grid):
    assert Collocated().resolve(grid) is Collocated().resolve(grid)


# ================================================================
#  ChebyshevMesh: the restricted (outer/Lobatto) family
# ================================================================
def test_chebyshev_collocated_resolves_to_lobatto(cheb_grid):
    mesh = cheb_grid.factors[0]
    assert Collocated().resolve(cheb_grid) is mesh.lobatto


def test_chebyshev_has_no_staggered_representation(cheb_grid):
    with pytest.raises(ValueError, match="STAGGERED"):
        Staggered("s").resolve(cheb_grid)


def test_chebyshev_profile_still_resolves(cheb_grid):
    mesh = cheb_grid.factors[0]
    assert Profile().resolve(cheb_grid) is mesh.constant
