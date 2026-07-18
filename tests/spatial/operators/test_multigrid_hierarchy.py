"""Tests for the multigrid hierarchy builder (B3, GM-D5).

The coarsening logic (semicoarsen the horizontal axes keeping the
vertical, or coarsen the vertical too under ``coarsen_vertical``; floor
at four cells, stop at indivisibility, cap at ``max_levels``, degrade to
one smoothing-only level on a tiny grid), the graceful degradation of a
non-coarsenable (Chebyshev) vertical, the ``rediscretize`` callback seam,
and the restricted-space-is-the-coarse-operator-space contract. The
model-specific FV re-discretization is exercised on the nonhydro2 side
(``tests/nonhydro2/test_pressure_fv.py``).
"""
from itertools import pairwise

import jax
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators import multigrid_hierarchy
from fridom.spatial.operators.multigrid_hierarchy import (
    MIN_COARSE_CELLS,
    _coarsenable_factors,
    _should_agglomerate,
    coarsen_levels,
    validate_agglomerate,
)
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


def mapped_z_grid(nx, ny, nz):
    """Build a nodal grid with a mapped (sigma) vertical."""
    sigma = MappedIntervalMesh(nz, (0.0, 1.0), lambda s: s ** 1.5,
                               periodic=False, name="z")
    meshes = (
        IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y"),
        sigma)
    grid = Grid(meshes)
    space = (grid.factors[0].center * grid.factors[1].center
             * grid.factors[2].center)
    return grid, space


def profile_grid(nx, ny, vertical_mesh):
    """Build an (x, y) nodal grid with a constant-z Profile space."""
    grid = Grid((
        IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y"),
        vertical_mesh))
    mx, my, mz = grid.factors
    return grid, mx.center * my.center * mz.constant


def shapes(levels):
    """Return the per-level operand shapes (finest first)."""
    return [tuple(space.shape) for (_grid, space, _t) in levels]


# ================================================================
#  Semicoarsening and the coarse-cell floor (default)
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


def test_none_coarsens_to_the_floor():
    # floor-limited depth (the default): 64 -> 32 -> 16 -> 8 -> 4 is a
    # five-level chain, halving each horizontal axis down to the floor
    grid, space = nodal_grid(64, 64, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=None)
    assert shapes(levels) == [
        (64, 64, 8), (32, 32, 8), (16, 16, 8), (8, 8, 8), (4, 4, 8)]
    assert levels[-1][2] is None


def test_none_is_the_default_argument():
    # omitting max_levels floors identically to an explicit None
    grid, space = nodal_grid(64, 64, 8)
    levels = coarsen_levels(grid, space, vertical="z")
    assert shapes(levels) == [
        (64, 64, 8), (32, 32, 8), (16, 16, 8), (8, 8, 8), (4, 4, 8)]


def test_none_degrades_to_one_level_on_a_tiny_grid():
    # floor-limited depth still yields a smoothing-only level when the
    # grid cannot coarsen at all (4x4 halves below the four-cell floor)
    grid, space = nodal_grid(4, 4, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=None)
    assert len(levels) == 1
    assert levels[0][2] is None


def test_uneven_horizontal_axes_coarsen_independently():
    # x can coarsen further than y: 16->8->4 while 8->4->(2 blocked)
    grid, space = nodal_grid(16, 8, 8)
    levels = coarsen_levels(grid, space, vertical="z", max_levels=5)
    assert shapes(levels) == [(16, 8, 8), (8, 4, 8), (4, 4, 8)]


# ================================================================
#  Full coarsening: coarsen_vertical (GM-D9 capability)
# ================================================================
def test_coarsen_vertical_coarsens_all_axes_uniform_z():
    # 64^3 -> 4^3 with the vertical coarsening under the same floor
    grid, space = nodal_grid(64, 64, 64)
    levels = coarsen_levels(grid, space, vertical="z",
                            coarsen_vertical=True)
    assert shapes(levels) == [
        (64, 64, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
    assert levels[-1][2] is None


def test_coarsen_vertical_on_a_mapped_z_grid():
    # the mapped (sigma) column coarsens just like the horizontals
    grid, space = mapped_z_grid(16, 16, 16)
    levels = coarsen_levels(grid, space, vertical="z",
                            coarsen_vertical=True)
    assert shapes(levels) == [(16, 16, 16), (8, 8, 8), (4, 4, 4)]


def test_coarsen_vertical_odd_nz_stops_but_horizontals_continue():
    # z: 20 -> 10 -> 5 (odd, indivisible) stops at 5; the horizontals
    # keep halving 64 -> 32 -> 16 -> 8 -> 4 past that point
    grid, space = nodal_grid(64, 64, 20)
    levels = coarsen_levels(grid, space, vertical="z",
                            coarsen_vertical=True)
    assert shapes(levels) == [
        (64, 64, 20), (32, 32, 10), (16, 16, 5), (8, 8, 5), (4, 4, 5)]


def test_coarsen_vertical_default_is_semicoarsening():
    # coarsen_vertical defaults False: the vertical stays full (MG-D4)
    grid, space = nodal_grid(16, 16, 16)
    levels = coarsen_levels(grid, space, vertical="z")
    assert shapes(levels) == [(16, 16, 16), (8, 8, 16), (4, 4, 16)]


def test_chebyshev_vertical_falls_back_without_error():
    # a Chebyshev vertical cannot coarsen (GM-D9 graceful degradation):
    # even with coarsen_vertical the z axis stays full, x/y still halve
    cheb = ChebyshevMesh(16, (0.0, 1.0), name="z")
    grid, space = profile_grid(16, 16, cheb)
    levels = coarsen_levels(grid, space, vertical="z",
                            coarsen_vertical=True)
    # the constant-z Profile factor keeps shape 1 throughout
    assert shapes(levels) == [(16, 16, 1), (8, 8, 1), (4, 4, 1)]
    assert levels[-1][2] is None


# ================================================================
#  vertical=None: no axis is designated vertical
# ================================================================
def test_vertical_none_designates_no_vertical():
    # None excludes no axis, so every axis coarsens (no coarsen_vertical
    # needed) — equivalent to full coarsening
    grid, space = nodal_grid(16, 16, 16)
    levels = coarsen_levels(grid, space, vertical=None)
    assert shapes(levels) == [(16, 16, 16), (8, 8, 8), (4, 4, 4)]


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
#  The rediscretize callback seam (model-agnostic here)
# ================================================================
def test_rediscretize_callback_runs_on_each_coarse_grid():
    grid, space = nodal_grid(16, 16, 8)
    seen = []
    coarsen_levels(grid, space, vertical="z", max_levels=3,
                   rediscretize=seen.append)
    # 3 levels -> 2 coarse grids built -> the callback fires twice, each
    # on a coarser grid (never the fine grid)
    assert len(seen) == 2
    assert all(g is not grid for g in seen)


def test_rediscretize_none_is_a_no_op():
    grid, space = nodal_grid(16, 16, 8)
    # the default rediscretize=None simply skips the callback
    levels = coarsen_levels(grid, space, vertical="z", max_levels=3)
    assert shapes(levels) == [(16, 16, 8), (8, 8, 8), (4, 4, 8)]


# ================================================================
#  Coarse-axis selection (unit)
# ================================================================
def test_coarsenable_factors_selects_horizontal_only():
    grid, _ = nodal_grid(16, 16, 8)
    assert _coarsenable_factors(grid, "z") == {"x": 2, "y": 2}


def test_coarsenable_factors_coarsen_vertical_includes_z():
    grid, _ = nodal_grid(16, 16, 8)
    assert _coarsenable_factors(grid, "z", coarsen_vertical=True) == {
        "x": 2, "y": 2, "z": 2}


def test_coarsenable_factors_vertical_none_includes_every_axis():
    grid, _ = nodal_grid(16, 16, 8)
    assert _coarsenable_factors(grid, None) == {"x": 2, "y": 2, "z": 2}


def test_coarsenable_factors_skips_non_coarsenable_vertical():
    cheb = ChebyshevMesh(16, (0.0, 1.0), name="z")
    grid, _ = profile_grid(16, 16, cheb)
    # a Chebyshev vertical is not coarsenable, so it is skipped even
    # under coarsen_vertical (the horizontals still qualify)
    assert _coarsenable_factors(grid, "z", coarsen_vertical=True) == {
        "x": 2, "y": 2}
    assert cheb.coarsenable is False


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
    # both skipped (they cannot coarsen)
    assert _coarsenable_factors(FakeGrid(), "z") == {}


# ================================================================
#  Coarse-grid agglomeration (MG-D10)
# ================================================================
def _sharded_coarse_extents(levels, device_count):
    """Per coarse level (index > 0), the min sharded per-shard extent.

    ``None`` marks a replicated (no sharded axis) coarse level.
    """
    out = []
    for grid, _space, _t in levels[1:]:
        axes = dict(grid.decomposition.default_layout.device_axes)
        if not axes:
            out.append(None)
            continue
        sizes = {name: mesh.n_cells
                 for mesh in grid.factors for name in mesh.names}
        out.append(min(-(-sizes[name] // device_count) for name in axes))
    return out


def test_validate_agglomerate_accepts_none_and_positive_int():
    assert validate_agglomerate(None) is None
    assert validate_agglomerate(1) == 1
    assert validate_agglomerate(4) == 4


def test_validate_agglomerate_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="positive integer"):
        validate_agglomerate(0)
    with pytest.raises(ValueError, match="positive integer"):
        validate_agglomerate(-2)


def test_validate_agglomerate_rejects_bool_and_float():
    with pytest.raises(TypeError, match="None or a positive integer"):
        validate_agglomerate(True)  # noqa: FBT003 — testing bool rejection
    with pytest.raises(TypeError, match="None or a positive integer"):
        validate_agglomerate(4.0)


def test_agglomerate_none_matches_the_default_chain():
    # agglomerate=None is the ordinary shard-or-replicate hierarchy:
    # identical shapes and grid identities to omitting the argument
    grid, space = nodal_grid(32, 32, 16)
    base = coarsen_levels(grid, space, vertical="z")
    same = coarsen_levels(grid, space, vertical="z", agglomerate=None)
    assert shapes(base) == shapes(same)
    assert all(a[0] is b[0] for a, b in zip(base, same, strict=True))


def test_agglomerate_preserves_floor_depth():
    # the switch replaces sharding with replication; it never changes
    # the number of levels (floor depth stays grid-size-driven)
    grid, space = nodal_grid(32, 32, 16)
    off = coarsen_levels(grid, space, vertical="z")
    on = coarsen_levels(grid, space, vertical="z", agglomerate=4)
    assert shapes(off) == shapes(on)


def test_agglomerate_is_a_noop_on_one_device():
    # on one device every level is already Layout({}); the knob changes
    # nothing (the mechanism reuses the natural replicated layout)
    if jax.device_count() != 1:
        pytest.skip("single-device semantics")
    grid, space = nodal_grid(32, 32, 16)
    off = coarsen_levels(grid, space, vertical="z")
    on = coarsen_levels(grid, space, vertical="z", agglomerate=4)
    assert shapes(off) == shapes(on)
    for _g, _s, _t in on:
        assert not dict(_g.decomposition.default_layout.device_axes)


@pytest.mark.multi_device
def test_agglomerate_replicates_below_the_switch():
    # the semicoarsened hierarchy keeps a divisible vertical, so its
    # coarse levels stay sharded at a tiny per-shard extent; agglomerate
    # replaces every sharded coarse level whose extent is below tau with
    # a replicated one, and leaves the above-threshold levels sharded
    p = jax.device_count()
    grid, space = nodal_grid(32, 32, 32)
    off = coarsen_levels(grid, space, vertical="z")
    on = coarsen_levels(grid, space, vertical="z", agglomerate=4)
    off_ext = _sharded_coarse_extents(off, p)
    on_ext = _sharded_coarse_extents(on, p)
    # OFF exposes at least one sharded coarse level below the threshold
    assert any(e is not None and e < 4 for e in off_ext), off_ext
    # ON has no sharded coarse level with extent below the threshold
    assert all(e is None or e >= 4 for e in on_ext), on_ext
    # once replicated, every deeper level stays replicated (monotone)
    first_repl = next(i for i, e in enumerate(on_ext) if e is None)
    assert all(e is None for e in on_ext[first_repl:])


@pytest.mark.multi_device
def test_agglomerate_transfer_across_switch_round_trips():
    # the restrict/prolong pair across the shard->replicate boundary is
    # finite (the all-gather / local-slice reshard the field factory
    # inserts must round-trip)
    grid, space = nodal_grid(32, 32, 32)
    levels = coarsen_levels(grid, space, vertical="z", agglomerate=4)
    for fine_grid, fine_space, transfer in levels[:-1]:
        f = fine_grid.create_field(fine_space)
        f = f.with_data(f.data + 1.0)
        r = transfer.restrict(f)
        p = transfer.prolong(r)
        assert bool(np.isfinite(np.asarray(r.data)).all())
        assert bool(np.isfinite(np.asarray(p.data)).all())


@pytest.mark.multi_device
def test_agglomerate_bytes_guard_keeps_large_levels_sharded(monkeypatch):
    # a level small enough per shard but too large to replicate stays
    # sharded (the conservative bytes guard); a one-byte guard blocks
    # every agglomeration, so the chain matches the un-agglomerated one
    monkeypatch.setattr(multigrid_hierarchy, "_AGGLOMERATE_MAX_BYTES", 1)
    grid, space = nodal_grid(32, 32, 32)
    off = coarsen_levels(grid, space, vertical="z")
    guarded = coarsen_levels(grid, space, vertical="z", agglomerate=4)
    p = jax.device_count()
    assert (_sharded_coarse_extents(off, p)
            == _sharded_coarse_extents(guarded, p))


def test_should_agglomerate_single_device_is_true():
    # on one device the switch always fires (Layout({}) either way)
    if jax.device_count() != 1:
        pytest.skip("single-device semantics")
    grid, _ = nodal_grid(8, 8, 8)
    coarse = grid.coarsened({"x": 2, "y": 2})
    assert _should_agglomerate(coarse, 4) is True
