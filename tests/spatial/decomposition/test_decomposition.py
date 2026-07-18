"""Tests for the Decomposition ABC, ReshardingReport, and SpaceLike."""
import dataclasses
from dataclasses import dataclass

import jax
import pytest

from fridom.spatial.decomposition.decomposition import (
    Decomposition,
    ReshardingReport,
    SpaceLike,
    check_level_shardability,
    negotiate,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


@dataclass(frozen=True)
class StandInSpace:

    """Minimal frozen stand-in implementing the SpaceLike protocol."""

    shape: tuple
    names: tuple
    layout: object = None

    @property
    def collapses_axis(self):
        # a full stand-in factor: not a collapsed (constant/trace) axis
        return False

    @property
    def factors(self):
        return (self,)

    def factor(self, name):
        if name in self.names:
            return self
        raise KeyError(name)


def test_decomposition_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Decomposition()


def test_abstract_surface_matches_class_doc():
    expected = {
        "halo", "default_layout", "layouts", "sharding",
        "local_slice", "storage_shape", "zeros", "pad", "unpad",
        "sync", "patch_physical_ends", "layout_for", "redistribute",
        "gather", "shard_writes", "chunk_hint",
    }
    assert set(Decomposition.__abstractmethods__) == expected


def test_stand_in_satisfies_space_protocol():
    space = StandInSpace(shape=(4,), names=("x",))
    assert isinstance(space, SpaceLike)
    assert space.factor("x") is space
    with pytest.raises(KeyError):
        space.factor("y")


def test_resharding_report_fields():
    old = Layout({})
    new = Layout({"x": "px"})
    report = ReshardingReport(old=old, new=new, changed=True)
    assert report.old == old
    assert report.new == new
    assert report.changed


def test_resharding_report_hashable_and_frozen():
    report = ReshardingReport(
        old=Layout({}), new=Layout({}), changed=False)
    same = ReshardingReport(
        old=Layout({}), new=Layout({}), changed=False)
    assert report == same
    assert hash(report) == hash(same)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.changed = True


# ================================================================
#  Replicated fallback (A3 / MG-D5)
# ================================================================
def test_allow_replicated_keeps_full_device_set():
    # 5 cells never shard over the forced 4 devices; with an explicit
    # device set the default raises, but allow_replicated keeps them
    # all under the replicated-only Layout({}).
    if jax.device_count() < 2:
        pytest.skip("needs several devices to request them")
    grid = Grid((IntervalMesh(5, (0.0, 1.0), name="x"),))
    ids = tuple(range(jax.device_count()))
    with pytest.raises(ValueError, match="GHOST-shardable"):
        negotiate(grid, grid.dispatch, device_ids=ids)
    decomp = negotiate(grid, grid.dispatch, device_ids=ids,
                       allow_replicated=True)
    assert decomp.layouts == (Layout({}),)
    assert decomp.device_count == jax.device_count()


def test_allow_replicated_does_not_change_auto_fallback():
    # auto-selection (device_ids=None) still falls back to one device,
    # independent of allow_replicated (the flag only relaxes the
    # explicit-device raise).
    grid = Grid((IntervalMesh(5, (0.0, 1.0), name="x"),))
    decomp = negotiate(grid, grid.dispatch, allow_replicated=True)
    assert decomp.layouts == (Layout({}),)
    assert decomp.device_count == 1


def test_grid_coarsened_replicates_below_the_floor():
    # a coarse level whose only axis no longer shards lives replicated
    # on the same device mesh (MG-D5), instead of raising.
    if jax.device_count() < 2:
        pytest.skip("needs several devices")
    ids = tuple(range(jax.device_count()))
    fine = Grid((IntervalMesh(16, (0.0, 1.0), name="x"),),
                device_ids=ids)
    assert dict(fine.decomposition.default_layout.device_axes) == {
        "x": "devices"}
    coarse = fine.coarsened(8)  # 2 cells: unshardable over 4 devices
    assert coarse.decomposition.layouts == (Layout({}),)
    assert (coarse.decomposition.device_count
            == fine.decomposition.device_count)


# ================================================================
#  Level-shardability gate (A5)
# ================================================================
def test_level_gate_passes_when_last_shard_covers_halo():
    check_level_shardability(
        (IntervalMesh(16, (0.0, 1.0), name="x"),),
        Layout({"x": "devices"}), HaloSpec({"x": 2}), 4,
        level="coarse level")


def test_level_gate_rejects_an_undersized_sharded_axis():
    # 8 cells over 4 devices: last shard holds 2, below halo(2) + 1
    with pytest.raises(ValueError, match="undersized"):
        check_level_shardability(
            (IntervalMesh(8, (0.0, 1.0), name="x"),),
            Layout({"x": "devices"}), HaloSpec({"x": 2}), 4,
            level="coarse level")


def test_level_gate_rejects_an_empty_trailing_shard():
    # 3 cells over 4 devices: last shard is empty (< 1 cell)
    with pytest.raises(ValueError, match="undersized"):
        check_level_shardability(
            (IntervalMesh(3, (0.0, 1.0), name="x"),),
            Layout({"x": "devices"}), HaloSpec({"x": 0}), 4)


def test_level_gate_names_the_level_and_axis():
    with pytest.raises(ValueError, match=r"coarse level.*'x'"):
        check_level_shardability(
            (IntervalMesh(8, (0.0, 1.0), name="x"),),
            Layout({"x": "devices"}), HaloSpec({"x": 2}), 4,
            level="coarse level")


def test_level_gate_is_a_noop_on_replicated_layout():
    # nothing sharded -> nothing to check, even on a tiny axis
    check_level_shardability(
        (IntervalMesh(3, (0.0, 1.0), name="x"),),
        Layout({}), HaloSpec({}), 4)


def test_level_gate_is_a_noop_on_a_single_device():
    check_level_shardability(
        (IntervalMesh(2, (0.0, 1.0), name="x"),),
        Layout({"x": "devices"}), HaloSpec({"x": 5}), 1)
