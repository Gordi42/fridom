"""Tests for the derived ``extra_halo`` helper (model/halo_demand)."""
from __future__ import annotations

import types

import pytest

import fridom as fr
from fridom.model.halo_demand import (
    derive_extra_halo,
    require_solver_halo,
    row_reach,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.meshes.interval import IntervalMesh


def _setup():
    """Return a plain periodic 2-D grid, its registry, and the x mesh."""
    mx = IntervalMesh(16, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(16, (0.0, 1.0), periodic=True, name="y")
    grid = fr.spatial.Grid((mx, my))
    return grid, mx


# ================================================================
#  row_reach — numbers straight off the bound operator
# ================================================================
def test_row_reach_is_two_sided():
    grid, mx = _setup()
    # the C-grid forward difference centre -> face reaches one cell
    # above, none below (a staggered stencil's asymmetry survives)
    assert row_reach(grid.dispatch, "diff", mx.center, "x") == (0, 1)
    # the backward difference face -> centre is the mirror image
    assert row_reach(grid.dispatch, "diff", mx.right, "x") == (1, 0)


# ================================================================
#  derive_extra_halo — barriers max, legs Minkowski-sum
# ================================================================
def test_barrier_takes_max_not_sum():
    grid, mx = _setup()
    # two barrier-separated legs, each a single order-2 difference:
    # the demands MAX to width 1, never sum to 2 (the projection rule)
    div = {"x": [("diff", mx.right)]}
    grad = {"x": [("diff", mx.center)]}
    spec = derive_extra_halo(grid.dispatch, ("x", "y"), [div, grad])
    assert spec["x"] == 1
    assert spec["y"] == 0  # a coordinate no leg touches stays reach 0


def test_within_a_leg_reaches_minkowski_sum():
    grid, mx = _setup()
    # two same-bias differences in ONE leg accumulate: 1 + 1 = 2
    leg = {"x": [("diff", mx.center), ("diff", mx.center)]}
    spec = derive_extra_halo(grid.dispatch, ("x",), [leg])
    assert spec["x"] == 2


def test_opposite_bias_pair_telescopes_within_a_leg():
    grid, mx = _setup()
    centre = mx.center
    face = grid.dispatch.resolve("diff", centre)["x"].codomain(centre)
    # a forward difference re-aligned by a backward interpolation:
    # [0, +1] (+) [-1, 0] = [-1, +1], width 1 (two-sided telescoping)
    leg = {"x": [("diff", centre), ("interpolate", face)]}
    spec = derive_extra_halo(grid.dispatch, ("x",), [leg])
    assert spec["x"] == 1


def test_result_is_symmetric():
    grid, mx = _setup()
    # a lone forward difference is (0, 1) two-sided; the declaration
    # collapses to the symmetric storage width max(0, 1) = 1
    spec = derive_extra_halo(
        grid.dispatch, ("x",), [{"x": [("diff", mx.center)]}])
    assert spec.interval("x") == (1, 1)


def test_no_legs_is_all_zero():
    grid, _ = _setup()
    spec = derive_extra_halo(grid.dispatch, ("x", "y"), [])
    assert spec["x"] == 0
    assert spec["y"] == 0


# ================================================================
#  require_solver_halo — the CG diagonal-builder guard
# ================================================================
def _halo_grid(widths):
    return types.SimpleNamespace(
        decomposition=types.SimpleNamespace(halo=HaloSpec(widths)))


def test_require_solver_halo_passes_when_wide_enough():
    # a width of 1 (or more) on every solved axis is sufficient
    require_solver_halo(
        _halo_grid({"x": 1, "z": 2}), ("x", "z"), solver="S")


def test_require_solver_halo_raises_when_too_narrow():
    with pytest.raises(ValueError, match="at least one negotiated ghost"):
        require_solver_halo(_halo_grid({"x": 0}), ("x",), solver="S")


def test_require_solver_halo_noop_without_decomposition():
    # a grid stand-in with no decomposition never raises (the check
    # is best-effort at solver build)
    require_solver_halo(types.SimpleNamespace(), ("x",), solver="S")
