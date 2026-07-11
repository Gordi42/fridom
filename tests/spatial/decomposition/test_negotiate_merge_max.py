"""Tests for the combined tendency + halo negotiation (merge_max)."""
import pytest

from fridom.spatial.decomposition.decomposition import (
    negotiate,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


@pytest.fixture
def grid():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")
    return Grid((mx, my))


@pytest.fixture
def space(grid):
    return grid.create_field().function_space


# ================================================================
#  tendency= and halo= combine per-coordinate max (merge_max)
# ================================================================
def test_extra_halo_wider_than_the_trace_wins(grid, space):
    decomp = negotiate(
        grid, grid.dispatch, state_spaces=(space,),
        tendency=lambda f: f.diff("x"),
        halo=HaloSpec({"x": 3}), device_ids=(0,))
    assert decomp.halo["x"] == 3
    assert decomp.halo["y"] == 0


def test_trace_wider_than_the_extra_halo_wins(grid, space):
    decomp = negotiate(
        grid, grid.dispatch, state_spaces=(space,),
        tendency=lambda f: f.diff("x"),
        halo=HaloSpec({"x": 0}), device_ids=(0,))
    assert decomp.halo["x"] == 1
    assert decomp.halo["y"] == 0


def test_mixed_per_coordinate_maximum(grid, space):
    def tendency(f):
        # parallel branches: the trace demands x:1 and y:1
        return f.diff("x"), f.diff("y")

    decomp = negotiate(
        grid, grid.dispatch, state_spaces=(space,),
        tendency=tendency, halo=HaloSpec({"x": 4}),
        device_ids=(0,))
    # x: extra (4) > trace (1); y: trace (1) > extra (absent -> 0)
    assert decomp.halo["x"] == 4
    assert decomp.halo["y"] == 1


def test_extra_halo_never_under_provisions_traced_names(grid, space):
    # an exclusive override would zero the traced x demand
    decomp = negotiate(
        grid, grid.dispatch, state_spaces=(space,),
        tendency=lambda f: f.diff("x"),
        halo=HaloSpec({"y": 2}), device_ids=(0,))
    assert decomp.halo["x"] == 1
    assert decomp.halo["y"] == 2


def test_tendency_with_halo_still_needs_state_spaces(grid):
    with pytest.raises(ValueError, match="state_spaces"):
        negotiate(grid, grid.dispatch, tendency=lambda f: f,
                  halo=HaloSpec({"x": 1}))


# ================================================================
#  Single-source paths keep their current meaning
# ================================================================
def test_halo_only_keeps_the_exclusive_meaning(grid):
    decomp = negotiate(grid, grid.dispatch, halo=HaloSpec({"x": 4}),
                       device_ids=(0,))
    assert decomp.halo["x"] == 4
    assert decomp.halo["y"] == 0


def test_tendency_only_is_the_pure_trace(grid, space):
    decomp = negotiate(
        grid, grid.dispatch, state_spaces=(space,),
        tendency=lambda f: f.diff("y"), device_ids=(0,))
    assert decomp.halo["x"] == 0
    assert decomp.halo["y"] == 1
