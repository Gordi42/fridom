"""Tests for the semicoarsened multigrid hierarchy builder (B3).

The degradation logic (semicoarsen the horizontal axes, keep the
vertical, floor at four cells, stop at indivisibility, cap at
``max_levels``, degrade to one smoothing-only level on a tiny grid),
the restricted-space-is-the-coarse-operator-space contract, and the
idempotent finite-volume re-discretization of the coarse grids.
"""
from itertools import pairwise

import numpy as np

from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.multigrid_hierarchy import (
    MIN_COARSE_CELLS,
    _coarsenable_factors,
    _is_fv,
    _rediscretize,
    coarsen_levels,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.transfer import GridTransfer

TWO_PI = 2.0 * np.pi


def nodal_grid(nx, ny, nz):
    """Build a plain nodal (Center) grid (x, y periodic, z bounded)."""
    meshes = (
        IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z"))
    grid = Grid(meshes)
    space = (grid.factors[0].center * grid.factors[1].center
             * grid.factors[2].center)
    return grid, space


def fv_immersed_grid(nx, ny, nz):
    """Build an FV (CellAvg) immersed grid, FV diff profile merged."""
    meshes = (
        IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z"))
    slope = lambda x, y, z: (z > 0.3).astype(float)  # noqa: ARG005, E731
    grid = Grid(meshes, immersed=ImmersedDomain(slope))
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    factors = [mesh.cell_avg for mesh in grid.factors]
    space = grid._laid_out(factors[0] * factors[1] * factors[2])
    return grid, space


def shapes(levels):
    """Return the per-level operand shapes (finest first)."""
    return [tuple(space.shape) for (_grid, space, _t) in levels]


# ================================================================
#  Semicoarsening and the coarse-cell floor
# ================================================================
def test_semicoarsening_keeps_the_vertical():
    grid, space = nodal_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    # x, y halve each level; z (the vertical) stays at 8 (MG-D4)
    assert shapes(levels) == [(16, 16, 8), (8, 8, 8), (4, 4, 8)]


def test_level_count_floors_at_four_cells():
    grid, space = nodal_grid(16, 16, 8)
    # max_levels is a MAXIMUM: 16 -> 8 -> 4 floors at four cells, so a
    # request for 5 still yields only 3 levels (2x2 is below the floor)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=5)
    assert shapes(levels) == [(16, 16, 8), (8, 8, 8), (4, 4, 8)]
    assert all(min(s[:2]) >= MIN_COARSE_CELLS for s in shapes(levels))


def test_degrades_to_one_level_on_a_tiny_grid():
    # 4x4: halving would land 2x2 < 4, so no coarsening is possible —
    # a one-level, smoothing-only hierarchy (must work, not raise)
    grid, space = nodal_grid(4, 4, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=5)
    assert len(levels) == 1
    assert levels[0][2] is None


def test_indivisibility_stops_coarsening():
    # 12 -> 6 (>= 4, ok) -> 3 (odd, indivisible) stops
    grid, space = nodal_grid(12, 12, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=5)
    assert shapes(levels) == [(12, 12, 8), (6, 6, 8)]


def test_max_levels_caps_the_depth():
    grid, space = nodal_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=2)
    assert shapes(levels) == [(16, 16, 8), (8, 8, 8)]


def test_uneven_horizontal_axes_coarsen_independently():
    # x can coarsen further than y: 16->8->4 while 8->4->(2 blocked)
    grid, space = nodal_grid(16, 8, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=5)
    assert shapes(levels) == [(16, 8, 8), (8, 4, 8), (4, 4, 8)]


# ================================================================
#  Transfers and the restricted-space contract
# ================================================================
def test_only_the_last_level_has_no_transfer():
    grid, space = nodal_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    assert all(isinstance(t, GridTransfer) for (_, _, t) in levels[:-1])
    assert levels[-1][2] is None


def test_restricted_space_is_the_next_operator_space():
    # the B2 contract: the transfer's restricted space IS the next
    # level's operator space (so the V-cycle never retags)
    grid, space = nodal_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    for (fine_grid, fine_space, transfer), (_cg, coarse_space, _t) in (
            pairwise(levels)):
        probe = fine_grid.create_field(fine_space)
        assert transfer.restrict(probe).function_space is coarse_space


# ================================================================
#  Coarse-axis selection (unit)
# ================================================================
def test_coarsenable_factors_selects_horizontal_only():
    grid, _ = nodal_grid(16, 16, 8)
    assert _coarsenable_factors(grid, "z") == {"x": 2, "y": 2}


def test_coarsenable_factors_honors_floor_and_parity():
    grid, _ = nodal_grid(4, 6, 8)  # 4 blocked (2<4), 6 -> 3 blocked
    assert _coarsenable_factors(grid, "z") == {}


def test_coarsenable_factors_skips_non_structured_and_multiname():
    class FakeMesh:
        def __init__(self, names):
            self.names = names

    class FakeGrid:
        factors = (FakeMesh(("q",)), FakeMesh(("a", "b")))

    # a non-StructuredMesh1D single-name mesh and a multi-name mesh are
    # both skipped (they cannot semicoarsen)
    assert _coarsenable_factors(FakeGrid(), "z") == {}


# ================================================================
#  Finite-volume re-discretization of the coarse grids (idempotent)
# ================================================================
def test_is_fv_detects_the_cell_average_family():
    _grid, fv_space = fv_immersed_grid(8, 8, 8)
    _ngrid, nodal_space = nodal_grid(8, 8, 8)
    assert _is_fv(fv_space) is True
    assert _is_fv(nodal_space) is False


def test_nodal_hierarchy_needs_no_rediscretization():
    grid, space = nodal_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    # a nodal coarse grid resolves diff on the registry defaults, so no
    # override profile is merged (override_keys stays empty)
    for coarse_grid, _s, _t in levels[1:]:
        assert coarse_grid.override_keys == frozenset()


def test_fv_coarse_grids_are_rediscretized():
    grid, space = fv_immersed_grid(16, 16, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    wanted = {*fv_cgrid_overrides(levels[1][0].factors)}
    # the coarse grid carries the FV diff profile (re-discretization)
    assert set(levels[1][0].override_keys) >= wanted


def test_rediscretize_is_idempotent():
    grid, _space = fv_immersed_grid(16, 16, 8)
    coarse = grid.coarsened({"x": 2, "y": 2})
    _rediscretize(coarse, fv=True)
    dispatch_after_first = coarse.dispatch
    # a second call is a no-op: the profile is merged exactly once, so
    # the registry object is unchanged (no growing layer stack)
    _rediscretize(coarse, fv=True)
    assert coarse.dispatch is dispatch_after_first


def test_rediscretize_skips_nodal_grids():
    grid, _space = nodal_grid(16, 16, 8)
    coarse = grid.coarsened({"x": 2, "y": 2})
    dispatch_before = coarse.dispatch
    _rediscretize(coarse, fv=False)
    assert coarse.dispatch is dispatch_before
    assert coarse.override_keys == frozenset()
