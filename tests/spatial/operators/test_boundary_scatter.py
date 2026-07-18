"""Tests for the boundary scatter operators (operators/boundary.py)."""
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.boundary import (
    BoundaryScatterAdd,
    BoundaryScatterSet,
)
from fridom.spatial.operators.verbs import scatter_add, scatter_set
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.trace import Side

N = 6


@pytest.fixture
def grid():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    mz = IntervalMesh(4, (0.0, 3.0), periodic=False, name="z")
    return Grid((mx, my, mz))


def _cell_space(grid):
    mx, my, mz = grid.factors
    return mx.center * my.center * mz.center


def _cell_field(grid, init):
    return grid.create_field(_cell_space(grid), init=init)


# ================================================================
#  Identity, interning, kind
# ================================================================
def test_scatter_variants_intern_and_differ():
    assert BoundaryScatterAdd() is BoundaryScatterAdd()
    assert BoundaryScatterSet() is BoundaryScatterSet()
    assert BoundaryScatterAdd() is not BoundaryScatterSet()
    assert BoundaryScatterAdd().dispatch_kind == "scatter_add"
    assert BoundaryScatterSet().dispatch_kind == "scatter_set"


def test_scatter_modes_differ_by_overwrite_flag():
    assert BoundaryScatterAdd()._overwrite is False
    assert BoundaryScatterSet()._overwrite is True


# ================================================================
#  codomain + loud rejections
# ================================================================
def test_scatter_codomain_is_the_full_space(grid):
    full = _cell_space(grid)
    trace = _cell_field(grid, lambda x, y, z: x + y + z).trace(
        "z", Side.HIGH).function_space.bare
    assert BoundaryScatterAdd().codomain(full, trace) is full


def test_scatter_rejects_extra_operands(grid):
    full = _cell_space(grid)
    trace = _cell_field(grid, lambda x, y, z: x + y + z).trace(
        "z", Side.HIGH).function_space.bare
    with pytest.raises(SpaceMismatchError, match="binary"):
        BoundaryScatterAdd().codomain(full, trace, full)


def test_scatter_rejects_a_non_trace_second_operand(grid):
    full = _cell_space(grid)
    with pytest.raises(SpaceMismatchError, match="exactly one"):
        BoundaryScatterAdd().codomain(full, full)


def test_scatter_rejects_a_node_set_mismatch(grid):
    mx, my, mz = grid.factors
    outer_trace = grid.create_field(
        mx.center * my.center * mz.outer,
        init=lambda x, y, z: x + y + z).trace(
            "z", Side.HIGH).function_space.bare
    # the full field is a Center, the trace parent is Outer
    with pytest.raises(SpaceMismatchError, match="parent node set"):
        BoundaryScatterAdd().codomain(_cell_space(grid), outer_trace)


def test_scatter_rejects_a_horizontal_mismatch(grid):
    mx, my, mz = grid.factors
    # the trace carries a Right(x) factor the full field does not
    trace = grid.create_field(
        mx.right * my.center * mz.center,
        init=lambda x, y, z: x + y + z).trace(
            "z", Side.HIGH).function_space.bare
    with pytest.raises(SpaceMismatchError, match="disagree"):
        BoundaryScatterAdd().codomain(_cell_space(grid), trace)


def test_scatter_side_aware_dirichlet_guard(grid):
    mx, my, mz = grid.factors
    # a BC-free trace at HIGH, but the target's Outer is Dirichlet-HIGH
    trace = grid.create_field(
        mx.center * my.center * mz.outer,
        init=lambda x, y, z: x + y + z).trace(
            "z", Side.HIGH).function_space.bare
    full = mx.center * my.center * mz.nodal(
        NodeSet.OUTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="HIGH wall"):
        BoundaryScatterAdd().codomain(full, trace)


def test_scatter_rejects_a_full_missing_the_traced_axis(grid):
    # a full field on a different mesh cannot resolve the traced axis
    mx, my, _mz = grid.factors
    trace = _cell_field(grid, lambda x, y, z: x + y + z).trace(
        "z", Side.HIGH).function_space.bare
    with pytest.raises(SpaceMismatchError, match="does not resolve"):
        BoundaryScatterAdd().codomain(mx.center * my.center, trace)


# ================================================================
#  Semantics + round-trips
# ================================================================
def test_scatter_add_into_zeros_equals_embed(grid):
    f = _cell_field(grid, lambda x, y, z: x + y + z)
    t = f.trace("z", Side.HIGH)
    scattered = scatter_add(grid.create_field(_cell_space(grid)), t)
    assert np.allclose(np.asarray(scattered.data),
                       np.asarray(t.embed("z").data))


def test_scatter_add_adds_into_the_boundary_row(grid):
    base = _cell_field(grid, lambda x, y, z: 5.0 + x + y + z)
    t = _cell_field(grid, lambda x, y, z: x + y + z).trace("z", Side.HIGH)
    out = scatter_add(base, t)
    expected = np.asarray(base.data).copy()
    expected[:, :, -1:] += np.asarray(t.data)
    assert np.allclose(np.asarray(out.data), expected)


def test_scatter_set_overwrites_the_boundary_row(grid):
    base = _cell_field(grid, lambda x, y, z: 5.0 + x + y + z)
    t = _cell_field(grid, lambda x, y, z: x + y + z).trace("z", Side.LOW)
    out = scatter_set(base, t)
    expected = np.asarray(base.data).copy()
    expected[:, :, :1] = np.asarray(t.data)
    assert np.allclose(np.asarray(out.data), expected)


def test_scatter_keeps_the_target_metadata(grid):
    base = _cell_field(grid, lambda x, y, z: x + y + z).with_metadata(name="q")
    t = _cell_field(
        grid, lambda x, y, z: 1.0 + x + y + z).trace("z", Side.HIGH)
    assert scatter_add(base, t).metadata.name == "q"


def test_scatter_on_a_stretched_mesh_is_a_pure_row_write():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    mz = MappedIntervalMesh(
        4, (0.0, 1.0), lambda s: s**2, periodic=False, name="z")
    grid = Grid((mx, mz))
    base = grid.create_field(mx.center * mz.center,
                             init=lambda x, z: x - z)
    t = base.trace("z", Side.HIGH)
    out = scatter_set(base, t)
    # overwriting a row with its own trace is the identity
    assert np.allclose(np.asarray(out.data), np.asarray(base.data))


def test_scatter_add_cellavg_stays_on_cellavg(grid):
    # the FV path: scatter into a CellAvg tendency stays CellAvg
    # (unlike embed, which lands the FV trace on co-located Center)
    mx, my, mz = grid.factors
    space = mx.center * my.center * mz.cell_avg
    tend = grid.create_field(space, init=lambda x, y, z: x + y + z)
    t = tend.trace("z", Side.HIGH)
    out = scatter_add(grid.create_field(space), t)
    assert out.function_space.bare is space
    data = np.asarray(out.data)
    assert np.allclose(data[:, :, -1:], np.asarray(t.data))
    assert np.allclose(data[:, :, :-1], 0.0)
