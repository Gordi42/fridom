"""Tests for the data-movement operators (Reshard, Sync)."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.movement import Reshard, Sync


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    # pinned to one device: these tests assert single-shard storage
    # semantics (the multi-device sync/reshard content lives in
    # decomposition/test_multi_device.py)
    return Grid((mx,), device_ids=(0,))


@pytest.fixture
def f(grid):
    return grid.create_field(init=lambda x: x**2, name="f")


# ================================================================
#  Sync (internal-only node)
# ================================================================
def test_sync_is_a_singleton():
    assert Sync() is Sync()


def test_sync_codomain_is_the_identity(grid):
    space = grid.create_field().function_space
    assert Sync().codomain(space) is space
    assert Sync().codomain(space, space) == (space, space)


def test_sync_delegates_to_the_grid(f):
    synced = Sync()(f)
    assert synced.function_space is f.function_space
    assert jnp.array_equal(synced.data, f.data)
    # the ghost slots are (re)filled: periodic wrap (width-aware —
    # the seeded registry's widest chain sets the negotiated width)
    w = f.grid.decomposition.halo["x"]
    assert jnp.array_equal(synced._data[:w], synced._data[-2 * w:-w])


def test_the_base_syncs_at_consumption(f):
    # consumption-side contract (task 1.8): the diff syncs its
    # operand (the exchange is memoized in the external identity
    # cache, NOT written onto the treedef-participating operand), and
    # the result's ghost slots are kernel-computed — valid to the
    # claimed depth, wrap-consistent there on this periodic mesh
    w = f.grid.decomposition.halo["x"]
    assert f.halo_valid["x"] == 0
    d = f.diff("x")
    assert f.halo_valid["x"] == 0  # operand treedef stable (direction a)
    valid = d.halo_valid["x"]
    assert valid == w - 1
    if valid:  # the claimed layers wrap like a synced field's
        assert jnp.array_equal(
            d._data[w - valid:w], d._data[-w - valid:-w])


# ================================================================
#  Reshard
# ================================================================
def test_reshard_is_interned_per_grid_and_target(grid):
    target = grid.decomposition.default_layout
    assert Reshard(grid, target) is Reshard(grid, target)


def test_reshard_rejects_targets_outside_the_vocabulary(grid):
    with pytest.raises(ValueError, match="vocabulary"):
        Reshard(grid, Layout({"x": "devices"}))


def test_reshard_codomain_attaches_the_target_layout(grid, mx):
    target = grid.decomposition.default_layout
    op = Reshard(grid, target)
    assert op.codomain(mx.center) is mx.center.with_layout(target)
    assert op.grid is grid
    assert op.target == target


def test_reshard_matching_layout_is_the_identity(grid, f):
    op = Reshard(grid, grid.decomposition.default_layout)
    result = op(f)
    # the kernel is skipped; the application still returns a synced
    # field on the same laid-out space
    assert result.function_space is f.function_space
    assert jnp.array_equal(result.data, f.data)


def test_reshard_sugar_elides_matching_layouts(f):
    assert f.reshard(f.function_space.layout) is f


def test_reshard_is_grid_bound(f):
    other = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    op = Reshard(other, other.decomposition.default_layout)
    with pytest.raises(GridMismatchError, match="grid-bound"):
        op(f)


def test_reshard_rejects_bare_spaces(grid, f, mx):
    op = Reshard(grid, grid.decomposition.default_layout)
    bare = ScalarField(grid, mx.center, f._data, f.metadata)
    with pytest.raises(SpaceMismatchError, match="bare"):
        op(bare)


def test_reshard_trace_reset_names(grid, mx):
    op = Reshard(grid, grid.decomposition.default_layout)
    # moved axes = names whose device assignment changes
    laid = mx.center.with_layout(Layout({"x": "devices"}))
    assert op._trace_reset_names(laid) == ("x",)
    same = mx.center.with_layout(Layout({}))
    assert op._trace_reset_names(same) == ()
