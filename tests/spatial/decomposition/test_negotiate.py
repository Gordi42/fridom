"""Tests for the negotiation entry point (decomposition.negotiate)."""
import jax
import pytest

from fridom.spatial.decomposition.decomposition import (
    _cap_for_sharding,
    _registry_halo,
    _shard_rank,
    _shardable_names,
    negotiate,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


# ================================================================
#  Halo sources: explicit > traced > registry maximum
# ================================================================
def test_explicit_halo_wins_and_missing_names_are_zero(grid):
    decomp = negotiate(grid, grid.dispatch, halo=HaloSpec({"x": 4}),
                       device_ids=(0,))
    assert decomp.halo["x"] == 4
    assert decomp.halo["y"] == 0


def test_two_sided_accounting_tightens_the_biased_chain(grid):
    # acceptance gate (perf/halo-interval): two-sided accounting
    # negotiates the true composed offset window, not the scalar sum
    # of symmetric halos. A wide biased stencil (FD order 6,
    # Center -> Right window [-2,+3], halo 3) composed with a narrow
    # flux-difference-like Right -> Center [-1,0] gives [-3,+3] =
    # width 3 -- where scalar accounting summed 3 + 1 = 4. This is the
    # same 4 -> 3 tightening upwind5/weno5 get (storage n+6, not n+8).
    space = grid.create_field().function_space
    fd6, fd2 = FiniteDifference(order=6), FiniteDifference(order=2)
    assert fd6.requirements(space.bare.factor("x")).reach == (2, 3)

    def tendency(state):
        return fd2["x"](fd6["x"](state))

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 3


def test_two_sided_accounting_tightens_the_centered_chain(grid):
    # the width-1 gate: a centered / diffusion chain, Center -> Right
    # [0,+1] then Right -> Center [-1,0], composes to [-1,+1] = width 1
    # (scalar sum 2). This is the FV order-2 derivative and the nodal
    # Laplacian tightening -- the "centered -> 1" acceptance value.
    space = grid.create_field().function_space
    fd2 = FiniteDifference(order=2)

    def tendency(state):
        return fd2["x"](fd2["x"](state))

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 1


def test_registry_maximum_is_the_provisional_default(grid):
    decomp = negotiate(grid, grid.dispatch, device_ids=(0,))
    # the widest seeded entry is the FV-derivative chain (reconstruct
    # [0,+1] then flux_diff [-1,0]); two-sided accounting composes them
    # to the true window [-1,+1] = width 1 (not the scalar sum 2)
    assert decomp.halo["x"] == 1
    assert decomp.halo["y"] == 1


def test_traced_tendency_overrides_the_registry_maximum(grid):
    space = grid.create_field().function_space

    def tendency(state):
        return state.diff("y")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 0
    # bounded Center -> Inner shrinks the codomain (8 -> 7 cells): the
    # exterior reach is 0 at the wall, but the per-shard footprint is
    # 1 (a sharded interior boundary reads one neighbor slot)
    assert decomp.halo["y"] == 1


def test_tendency_without_state_spaces_raises(grid):
    with pytest.raises(ValueError, match="state_spaces"):
        negotiate(grid, grid.dispatch, tendency=lambda state: state)


def test_tendency_and_halo_combine_as_merge_max(grid):
    # work item 1 (phase-2 reconciliation): extra_halo bypass demand
    # merges with the trace, neither shadows the other
    space = grid.create_field().function_space

    def tendency(state):
        return state.diff("y")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency,
                       halo=HaloSpec({"x": 3}), device_ids=(0,))
    assert decomp.halo["x"] == 3
    # bounded Center -> Inner: exterior reach is 0 at the wall, but the
    # per-shard footprint is 1 (a sharded interior boundary reads one
    # neighbor slot)
    assert decomp.halo["y"] == 1


def test_traced_chains_accumulate_the_sync_free_demand(grid):
    # task 1.8: the trace records the sync-free width of the step — a
    # triple diff chain on the periodic axis. Two-sided accounting:
    # the alternating windows [0,+1], [-1,0], [0,+1] compose to
    # [-1,+2], width 2 (not the scalar sum 3)
    space = grid.create_field().function_space

    def tendency(state):
        return state.diff("x").diff("x").diff("x")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 2
    assert decomp.halo["y"] == 0


def test_traced_bounded_chains_demand_the_per_application_max(
        grid, my):
    # bounded axes re-sync at every stencil (kernel claims reset
    # there), so the sync-free demand is the per-application max, not
    # the sum. The chain runs Outer -> Center -> Inner; both hops
    # shrink the codomain (9 -> 8 -> 7 cells), so each difference's
    # exterior reach cancels to 0 at the wall — but its per-shard
    # stencil footprint is 1 (a sharded interior boundary reads a
    # neighbor slot), so the per-application max is 1
    space = grid.create_field().function_space.bare.replace(
        y=my.outer)

    def tendency(state):
        return state.diff("y").diff("y")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["y"] == 1


def test_arithmetic_resets_the_traced_demand(grid):
    # +/- re-store (zero validity), so depth never accumulates
    # through them: two single-diff segments, not one depth-2 chain
    space = grid.create_field().function_space

    def tendency(state):
        return (2.0 * state.diff("x").to(space.bare)).diff("x")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    # each segment is a diff [0,+1] then interp [-1,0] = window
    # [-1,+1], width 1; the +/- reset keeps them from summing
    assert decomp.halo["x"] == 1


def test_cap_for_sharding_lowers_wide_traces_to_the_shard_extent(
        mx):
    # 8 cells over 4 devices: cells/shard = 2, cap = 1
    spec = HaloSpec({"x": 5})
    floor = HaloSpec({"x": 1})
    capped = _cap_for_sharding((mx,), spec, floor, 4)
    assert capped["x"] == 1
    # a floor that does not fit keeps its width (fails as before)
    wide_floor = HaloSpec({"x": 3})
    kept = _cap_for_sharding((mx,), HaloSpec({"x": 5}), wide_floor, 4)
    assert kept["x"] == 5
    # nothing to cap: the spec object passes through
    assert _cap_for_sharding((mx,), floor, floor, 4) is floor
    # heavy padding is skipped (no cap derivable): 8 over 5 gives
    # cells = ceil(8/5) = 2, last = 8 - 4*2 = 0 (< 1)
    assert _cap_for_sharding((mx,), spec, floor, 5) is spec
    # a mild non-divisible axis still caps to last - 1: 8 over 3
    # gives cells = 3, last = 2, cap = 1
    assert _cap_for_sharding((mx,), spec, floor, 3)["x"] == 1
    # a floor that never saw the name counts as zero
    assert _cap_for_sharding((mx,), spec, HaloSpec({}), 4)["x"] == 1


def test_cap_for_sharding_on_non_divisible_axes():
    # P = 4; cap = last - 1, which reduces to cells - 1 on a
    # divisible axis. Called directly (device-count independent).
    floor = HaloSpec({"x": 1})
    # 257 cells: cells = ceil(257/4) = 65, last = 257 - 3*65 = 62,
    # so a wide demand caps to last - 1, i.e. 61
    m257 = IntervalMesh(257, (0.0, 1.0), name="x")
    wide = HaloSpec({"x": 100})
    assert _cap_for_sharding((m257,), wide, floor, 4)["x"] == 61
    # 5 cells: heavy padding (cells = 2, last = -1) -> not capped,
    # spec passes through unchanged
    m5 = IntervalMesh(5, (0.0, 1.0), name="x")
    assert _cap_for_sharding((m5,), wide, floor, 4) is wide
    # divisible regression: 16 cells, cells = 4, last = 4,
    # cap = 3 == n_cells // 4 - 1 (the pre-change value)
    m16 = IntervalMesh(16, (0.0, 1.0), name="x")
    assert _cap_for_sharding((m16,), HaloSpec({"x": 5}), floor, 4)[
        "x"] == 3


def test_registry_halo_scopes_to_state_space_meshes(grid, my):
    # scoping to a y-only state silences the x-mesh demands (the y
    # width is 1: the two-sided FV-derivative chain is the widest
    # seeded entry, composed window [-1,+1])
    spec = _registry_halo(("x", "y"), grid.dispatch,
                          state_spaces=(my.center,))
    assert spec["x"] == 0
    assert spec["y"] == 1


def test_registry_halo_without_items_surface_is_zero():
    spec = _registry_halo(("x",), object())
    assert spec["x"] == 0


# ================================================================
#  Layout selection (device-count independent pieces)
# ================================================================
def test_single_device_negotiation_is_the_trivial_layout(grid):
    decomp = negotiate(grid, grid.dispatch, device_ids=(0,))
    assert isinstance(decomp, TensorDecomposition)
    assert decomp.layouts == (Layout({}),)


def test_shardable_names_admit_mild_and_reject_heavy(mx, my):
    halo = HaloSpec({"x": 1, "y": 1})
    # 8 cells over 4 devices: divisible, last = 2 >= halo + 1
    assert _shardable_names((mx, my), halo, 4) == ("x", "y")
    # 8 cells over 3 devices: mild non-divisible (cells = 3,
    # last = 8 - 2*3 = 2), so both axes stay shardable
    assert _shardable_names((mx, my), halo, 3) == ("x", "y")
    # 8 cells over 5 devices: heavy padding (cells = 2, last = 0),
    # trailing shards empty -> both meshes rejected
    assert _shardable_names((mx, my), halo, 5) == ()


def test_shardable_names_non_divisible_concrete_cases():
    # P = 4, min_local = 1 for the interval ghost families. Called
    # directly (device-count independent, no real devices needed).
    # 257 cells: cells = ceil(257/4) = 65, last = 257 - 3*65 = 62,
    # shardable for width 1 (last >= width + 1 = 2)
    m257 = IntervalMesh(257, (0.0, 1.0), name="x")
    assert _shardable_names((m257,), HaloSpec({"x": 1}), 4) == ("x",)
    # 13 cells: cells = 4, last = 13 - 3*4 = 1. The shortest shard
    # holds a single cell -- below width + 1 = 2, so width-1
    # sharding is rejected, but a width-0 exchange edge fits
    m13 = IntervalMesh(13, (0.0, 1.0), name="x")
    assert _shardable_names((m13,), HaloSpec({"x": 1}), 4) == ()
    assert _shardable_names((m13,), HaloSpec({"x": 0}), 4) == ("x",)
    # 5 cells: cells = ceil(5/4) = 2, last = 5 - 3*2 = -1 -> heavy
    # padding (trailing shards empty), rejected for any width
    m5 = IntervalMesh(5, (0.0, 1.0), name="x")
    assert _shardable_names((m5,), HaloSpec({"x": 0}), 4) == ()
    # divisible regression: 16 cells, cells = 4, last = 4 == cells,
    # so the check is identical to the pre-change behavior
    m16 = IntervalMesh(16, (0.0, 1.0), name="x")
    assert _shardable_names((m16,), HaloSpec({"x": 1}), 4) == ("x",)
    assert _shardable_names((m16,), HaloSpec({"x": 4}), 4) == ()


def test_shardable_names_respect_the_halo_constraint(mx):
    # 2 cells/shard must cover halo + 1 (short staggered last shard)
    assert _shardable_names((mx,), HaloSpec({"x": 2}), 4) == ()
    assert _shardable_names((mx,), HaloSpec({"x": 1}), 4) == ("x",)


def test_shardable_names_skip_meshes_without_ghost_traits():
    class NoTraitsMesh:
        n_cells = 8
        names = ("q",)

    assert _shardable_names((NoTraitsMesh(),),
                            HaloSpec({"q": 0}), 4) == ()


# ================================================================
#  Shardable-name ordering (staggering-aware default selection)
# ================================================================
def test_shard_rank_classifies_by_periodicity_and_divisibility():
    # rank 0: periodic + divisible; rank 1: walled + divisible;
    # rank 2: indivisible (periodicity irrelevant)
    periodic = IntervalMesh(16, (0.0, 1.0), name="x")
    walled = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    indivisible = IntervalMesh(23, (0.0, 1.0), name="x")
    assert _shard_rank(periodic, 16, 4) == 0
    assert _shard_rank(walled, 16, 4) == 1
    assert _shard_rank(indivisible, 23, 4) == 2

    # a mesh that does not declare periodicity is treated
    # conservatively: rank 1 when divisible, rank 2 when not
    class NoPeriodicity:
        pass

    assert _shard_rank(NoPeriodicity(), 16, 4) == 1
    assert _shard_rank(NoPeriodicity(), 23, 4) == 2


def test_shardable_order_prefers_periodic_divisible_over_walled():
    # both divisible by 4; the periodic axis keeps every field extent
    # divisible, so it ranks ahead of the walled one even though the
    # walled x is first in grid order
    mx = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")  # walled
    my = IntervalMesh(16, (0.0, 2.0), name="y")                  # periodic
    halo = HaloSpec({"x": 1, "y": 1})
    assert _shardable_names((mx, my), halo, 4) == ("y", "x")


def test_shardable_order_prefers_divisible_over_indivisible():
    # walled-divisible (rank 1) beats periodic-indivisible (rank 2): a
    # divisible cell count keeps the reblock collective-free, while an
    # indivisible one pays the padded-even reblock on every excursion
    mx = IntervalMesh(23, (0.0, 1.0), name="x")  # periodic, 23 % 4 != 0
    my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")  # walled
    halo = HaloSpec({"x": 1, "y": 1})
    assert _shardable_names((mx, my), halo, 4) == ("y", "x")


def test_shardable_order_grid_order_breaks_ties_within_a_rank():
    # two periodic divisible axes stay in grid order (rank-0 tie), so a
    # fully-periodic divisible grid keeps its first axis as the default
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    mz = IntervalMesh(16, (0.0, 3.0), name="z")
    halo = HaloSpec({"x": 1, "y": 1, "z": 1})
    assert _shardable_names((mx, my, mz), halo, 4) == ("x", "y", "z")
    # all walled divisible -> rank-1 tie -> grid order preserved
    wx = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    wy = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    assert _shardable_names((wx, wy), halo, 4) == ("x", "y")
    # both indivisible -> rank-2 tie -> grid order (no residue overfit)
    ix = IntervalMesh(23, (0.0, 1.0), name="x")  # 23 % 4 == 3
    iy = IntervalMesh(17, (0.0, 2.0), name="y")  # 17 % 4 == 1
    assert _shardable_names((ix, iy), halo, 4) == ("x", "y")


def test_shardable_order_single_axis_is_unchanged():
    # one qualifying axis: the order is trivially itself, walled or not
    walled = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    assert _shardable_names((walled,), HaloSpec({"x": 1}), 4) == ("x",)


def test_shardable_order_reorders_but_keeps_the_full_set():
    # reordering never adds or drops a name: the walled-x + periodic-y
    # grid still offers both pencils, only the default (first) changes
    mx = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    halo = HaloSpec({"x": 1, "y": 1})
    names = _shardable_names((mx, my), halo, 4)
    assert set(names) == {"x", "y"}
    assert names[0] == "y"  # periodic default


def test_shardable_order_heavy_padding_still_rejected():
    # ordering does not resurrect a rejected axis: 8 cells over 5
    # devices empties a trailing shard, so neither axis qualifies
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")
    assert _shardable_names((mx, my), HaloSpec({"x": 0, "y": 0}), 5) == ()


def test_explicit_devices_with_nothing_shardable_raise():
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")  # 5 % 4 != 0
    grid = Grid((mesh,))
    if jax.device_count() < 2:
        pytest.skip("needs several devices to request them")
    ids = tuple(range(jax.device_count()))
    with pytest.raises(ValueError, match="GHOST-shardable"):
        negotiate(grid, grid.dispatch, device_ids=ids)


def test_auto_selection_falls_back_to_one_device():
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")  # never divisible
    grid = Grid((mesh,))
    assert grid.decomposition.layouts == (Layout({}),)
    assert grid.decomposition._device_mesh.size == 1


# ================================================================
#  Multi-device negotiation (forced-devices suite)
# ================================================================
@pytest.mark.multi_device
def test_default_layout_shards_the_first_ghost_factor():
    # 16 cells: shardable across 4 devices under the halo-2
    # constraint (cells/shard >= halo + 1) of the seeded registry
    mx16 = IntervalMesh(16, (0.0, 1.0), name="x")
    my16 = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    decomp = Grid((mx16, my16)).decomposition
    default = decomp.default_layout
    assert dict(default.device_axes) == {"x": "devices"}
    # pencil for the y factor plus the replicated fallback
    assert Layout({"y": "devices"}) in decomp.layouts
    assert decomp.layouts[-1] == Layout({})
    assert decomp._device_mesh.size == jax.device_count()


@pytest.mark.multi_device
def test_default_layout_shards_a_mild_non_divisible_factor():
    # a non-divisible cell count (device_count * 64 + 1) is mild
    # padding (cells = ceil, last >= 1), so x now negotiates a
    # sharded default layout instead of falling back to one device.
    # Only the SELECTED layout is inspected -- no storage is
    # materialized on the non-divisible grid.
    n = jax.device_count() * 64 + 1
    grid = Grid((IntervalMesh(n, (0.0, 1.0), name="x"),))
    ids = tuple(range(jax.device_count()))
    decomp = negotiate(grid, grid.dispatch, device_ids=ids)
    assert dict(decomp.default_layout.device_axes) == {"x": "devices"}
    assert decomp._device_mesh.size == jax.device_count()


@pytest.mark.multi_device
def test_default_layout_dodges_a_walled_axis_for_a_periodic_one():
    # walled x + periodic y, both divisible: the staggering-aware order
    # shards the periodic y so the walled deficit leg never lands on the
    # storage-shard axis. The walled x is still an available pencil.
    n = jax.device_count() * 4
    mx = IntervalMesh(n, (0.0, 1.0), periodic=False, name="x")
    my = IntervalMesh(n, (0.0, 2.0), name="y")
    decomp = Grid((mx, my)).decomposition
    assert dict(decomp.default_layout.device_axes) == {"y": "devices"}
    assert Layout({"x": "devices"}) in decomp.layouts
    assert decomp._device_mesh.size == jax.device_count()


@pytest.mark.multi_device
def test_default_layout_shards_the_walled_axis_when_alone():
    # fully-walled xyz: no periodic (rank-0) axis qualifies, so the
    # grid-first walled axis is still the default -- behaviour unchanged
    # from grid-order selection (there is no escape from the wall).
    n = jax.device_count() * 4
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 + i), periodic=False, name=nm)
        for i, nm in enumerate(("x", "y", "z")))
    decomp = Grid(meshes).decomposition
    assert dict(decomp.default_layout.device_axes) == {"x": "devices"}


@pytest.mark.multi_device
def test_fully_periodic_divisible_keeps_grid_order_default():
    # the byte-identity guarantee: a fully-periodic divisible grid is
    # all rank 0, so grid order is preserved -- the default shards x and
    # the full pencil vocabulary matches grid-order selection, so the
    # compiled program is unchanged from before the reordering.
    n = jax.device_count() * 4
    meshes = tuple(IntervalMesh(n, (0.0, 1.0 + i), name=nm)
                   for i, nm in enumerate(("x", "y", "z")))
    decomp = Grid(meshes).decomposition
    assert dict(decomp.default_layout.device_axes) == {"x": "devices"}
    assert decomp.layouts == (
        Layout({"x": "devices"}), Layout({"y": "devices"}),
        Layout({"z": "devices"}), Layout({}))


@pytest.mark.multi_device
def test_default_layout_shards_indivisible_only_when_alone():
    # an indivisible periodic axis (rank 2) loses to a divisible walled
    # axis (rank 1): the divisible axis reblocks collective-free, so it
    # is the better default despite the wall.
    nx = jax.device_count() * 4 + 1  # indivisible
    ny = jax.device_count() * 4      # divisible
    mx = IntervalMesh(nx, (0.0, 1.0), name="x")                  # periodic
    my = IntervalMesh(ny, (0.0, 2.0), periodic=False, name="y")  # walled
    decomp = Grid((mx, my)).decomposition
    assert dict(decomp.default_layout.device_axes) == {"y": "devices"}


@pytest.mark.multi_device
def test_layout_for_reaches_a_pencil(grid):
    decomp = grid.decomposition
    pencil = decomp.layout_for(("x",))
    assert pencil.is_local("x")
