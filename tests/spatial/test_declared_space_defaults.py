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

from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import (
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
    from fridom.spatial.spaces.nodal import (  # noqa: PLC0415
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
    from fridom.spatial.spaces.nodal import (  # noqa: PLC0415
        NodeSet,
    )
    x, z = meshes
    space = Collocated(bc={"z": BC.DIRICHLET}).resolve(grid)
    assert space.factors == (
        x.center, z.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))


def test_repeated_resolution_is_interned(grid):
    assert Collocated().resolve(grid) is Collocated().resolve(grid)


# ================================================================
#  The fv (average) family (FV-D1b / FV-D2 option A)
# ================================================================
def test_grid_default_family_is_nodal(grid):
    assert grid.default_family == "nodal"


def test_fv_collocated_resolves_to_cell_averages(grid, meshes):
    x, z = meshes
    space = Collocated(family="fv").resolve(grid)
    assert space.factors == (x.cell_avg, z.cell_avg)


def test_fv_staggered_stays_the_nodal_face(grid, meshes):
    # FV-D2 option A: a staggered coordinate keeps the point-value
    # face (Right periodic / Inner bounded); the collocated
    # coordinates land on CellAvg
    x, z = meshes
    assert Staggered("x", family="fv").resolve(grid).factors == (
        x.right, z.cell_avg)
    assert Staggered("z", family="fv").resolve(grid).factors == (
        x.cell_avg, z.inner)


def test_fv_profile_mixes_constant_and_cell_average(grid, meshes):
    x, z = meshes
    assert Profile("z", family="fv").resolve(grid).factors == (
        x.constant, z.cell_avg)


def test_grid_level_fv_default_resolves_cell_averages(meshes):
    x, z = meshes
    fv_grid = Grid(meshes, family="fv")
    assert fv_grid.default_family == "fv"
    assert Collocated().resolve(fv_grid).factors == (
        x.cell_avg, z.cell_avg)
    # a per-field family= override wins over the grid default
    assert Collocated(family="nodal").resolve(fv_grid).factors == (
        x.center, z.center)


def test_fv_collocated_bc_is_a_taught_error(grid):
    with pytest.raises(ValueError, match="BC on a family='fv'"):
        Collocated(bc={"x": BC.DIRICHLET},
                   family="fv").resolve(grid)


def test_fv_on_chebyshev_is_a_taught_error(cheb_grid):
    with pytest.raises(ValueError, match="ChebyshevMesh has no "
                       "cell averages"):
        Collocated(family="fv").resolve(cheb_grid)


def test_grid_level_family_validation(meshes):
    with pytest.raises(ValueError,
                       match="grid-level default family must be one"):
        Grid(meshes, family="bogus")


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
