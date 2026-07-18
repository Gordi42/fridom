"""Tests for freeze(): fingerprint, verify path, GridFrozenError."""
import jax
import pytest

from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import GridFrozenError
from fridom.spatial.grid import Grid, NegotiationFingerprint
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


def diff_x(state):
    return state.diff("x")


def no_op(state):
    return state


# ================================================================
#  Fingerprint content
# ================================================================
def test_fingerprint_is_none_before_freeze(grid):
    assert grid.fingerprint is None


def test_freeze_records_the_fingerprint(grid, mx):
    space = grid.create_field().function_space
    grid.merge_overrides({("diff", mx.center): FiniteDifference(
        order=2)})
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    fp = grid.fingerprint
    assert isinstance(fp, NegotiationFingerprint)
    assert fp.state_spaces == frozenset({space.bare})
    assert fp.override_keys == frozenset({("diff", mx.center)})
    assert fp.halo == grid.decomposition.halo
    assert fp.halo["x"] == 1
    assert fp.halo["y"] == 0
    assert fp.layouts == grid.decomposition.layouts


def test_provisional_freeze_records_empty_demands(grid):
    grid.freeze()
    fp = grid.fingerprint
    assert fp.state_spaces == frozenset()
    assert fp.override_keys == frozenset()
    assert fp.halo == grid.decomposition.halo


def test_freeze_is_idempotent(grid):
    grid.freeze()
    fp = grid.fingerprint
    grid.freeze()
    assert grid.fingerprint is fp


# ================================================================
#  Verify path: satisfaction passes (subset / less-or-equal)
# ================================================================
def test_verify_passes_on_equal_demand(grid):
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    report = grid.negotiate(state_spaces=(space,), tendency=diff_x)
    assert report.changed is False
    assert report.old is report.new
    assert report.new is grid.decomposition.default_layout


def test_verify_passes_on_strictly_smaller_demand(grid):
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    # zero traced demand < frozen x-halo of 1: satisfaction, not
    # equality
    report = grid.negotiate(state_spaces=(space,), tendency=no_op)
    assert report.changed is False


def test_verify_passes_on_smaller_extra_halo(grid):
    # frozen provisionally: halo = registry per-operator maximum
    grid.freeze()
    frozen_x = grid.fingerprint.halo["x"]
    report = grid.negotiate(halo=HaloSpec({"x": frozen_x}))
    assert report.changed is False


# ================================================================
#  Verify path: larger demands raise GridFrozenError
# ================================================================
def test_verify_fails_on_larger_extra_halo(mx, my):
    # a larger extra halo than the frozen record fails. Pin the grid
    # to one device so the shardable-cap never absorbs the demand (it
    # would on a sharded grid: width above the floor is satisfiable via
    # re-sync); the raw comparison then applies at any device count.
    grid = Grid((mx, my), device_ids=(0,))
    grid.freeze()
    frozen_x = grid.fingerprint.halo["x"]
    with pytest.raises(GridFrozenError,
                       match=rf"halo\['x'\]: demanded "
                             rf"{frozen_x + 2} > frozen {frozen_x}"):
        grid.negotiate(halo=HaloSpec({"x": frozen_x + 2}))


def test_verify_merges_extra_halo_with_the_trace(mx, my):
    # merge_max rule: halo= widens the traced demand, it never
    # overrides it — the extra spec alone must be able to fail. Pin to
    # one device so the cap never absorbs the extra halo.
    grid = Grid((mx, my), device_ids=(0,))
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    with pytest.raises(GridFrozenError, match=r"halo\['x'\]"):
        grid.negotiate(state_spaces=(space,), tendency=diff_x,
                       halo=HaloSpec({"x": 3}))


def test_verify_failure_lists_every_violation(grid, mx, my):
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    foreign = mx.right * my.center
    with pytest.raises(GridFrozenError,
                       match=r"(?s)halo\['y'\].*not in the frozen "
                             "record"):
        grid.negotiate(state_spaces=(foreign,), tendency=no_op,
                       halo=HaloSpec({"y": 2}))


def test_frozen_negotiate_never_renegotiates(grid):
    grid.freeze()
    decomposition = grid.decomposition
    grid.negotiate()
    assert grid.decomposition is decomposition


# ================================================================
#  Verify path: shardable-cap symmetry (task 1.8)
# ================================================================
#  On a sharded grid the negotiate path lowers a wide sync-free demand
#  to the shortest-shard extent (``_cap_for_sharding``) before freeze
#  records it, so the verify path must apply the SAME cap or an
#  identical re-assembly (``Model.variant``) would spuriously fault.
#  The scenario drives the cap with a large explicit ``halo=`` on top
#  of a tendency: the tendency keeps negotiate and verify on the
#  symmetric traced base, and a width far above the shard extent makes
#  the cap bite regardless of the exact per-operator halo accounting.
def _capped_shard_grid():
    # nx = 4 * device_count keeps the shortest (only) shard 4 cells
    # for any device count >= 2, so cap = last - 1 = 3 and x still
    # shards (last 4 >= capped 3 + 1). Scaling with the device count
    # keeps the cap engaged under both a 2- and a forced-4 backend.
    nx = 4 * jax.device_count()
    mx = IntervalMesh(nx, (0.0, 1.0), name="x")             # periodic
    my = IntervalMesh(nx, (0.0, 2.0), periodic=False, name="y")
    return Grid((mx, my))


#: an extra halo demand far above the shortest-shard cap
WIDE = HaloSpec({"x": 10})


@pytest.mark.multi_device
def test_verify_caps_the_reassembly_demand_like_negotiate():
    # the reported bug: cap on write, no cap on verify -> a
    # byte-identical re-assembly on the frozen grid faults
    grid = _capped_shard_grid()
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x, halo=WIDE)
    # the cap engaged: the width-10 demand recorded as the shard
    # extent 3, and x is genuinely sharded (else the scenario is
    # vacuous)
    assert grid.decomposition.device_count == jax.device_count()
    assert dict(grid.decomposition.default_layout.device_axes) == {
        "x": "devices"}
    grid.freeze()
    assert grid.fingerprint.halo["x"] == 3
    # the identical negotiation now verifies capped-vs-capped instead
    # of raw-10 > frozen-3 (which raised before the symmetric cap)
    report = grid.negotiate(state_spaces=(space,), tendency=diff_x,
                            halo=WIDE)
    assert report.changed is False


@pytest.mark.multi_device
def test_verify_cap_still_rejects_a_genuine_violation():
    # the cap never lowers a demand below the shard extent (and never
    # below the per-application floor), so a genuine over-demand on an
    # axis the frozen grid never provisioned (frozen y = 0) still
    # raises -- the width-10 extra halo is lowered only to the shard
    # extent 3, which still exceeds the recorded 0
    grid = _capped_shard_grid()
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x, halo=WIDE)
    grid.freeze()
    assert grid.fingerprint.halo["y"] == 0
    with pytest.raises(GridFrozenError,
                       match=r"halo\['y'\]: demanded 3 > frozen 0"):
        grid.negotiate(state_spaces=(space,), tendency=diff_x,
                       halo=HaloSpec({"y": 10}))


@pytest.mark.multi_device
def test_verify_accepts_a_larger_extra_halo_that_caps_to_record():
    # negotiate-consistent capped acceptance: on a cap-engaged sharded
    # grid an extra halo even larger than the frozen one verifies
    # cleanly, because both cap to the shard extent -- width above the
    # per-application floor is satisfiable via runtime re-sync, so it
    # is not a violation (this is the asymmetry the fix removes)
    grid = _capped_shard_grid()
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x, halo=WIDE)
    grid.freeze()
    assert grid.fingerprint.halo["x"] == 3
    report = grid.negotiate(state_spaces=(space,), tendency=diff_x,
                            halo=HaloSpec({"x": 20}))
    assert report.changed is False


def test_verify_never_caps_on_a_single_device():
    # the cap is conditional on a genuinely sharded frozen grid: a
    # single-device grid holds the raw demand, so a larger demand
    # must fault at its true width and never be silently lowered to
    # the (single-shard) cap. Guards the ``device_count > 1`` gate.
    grid = Grid((IntervalMesh(4, (0.0, 1.0), name="x"),),
                device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    grid.freeze()
    assert grid.decomposition.device_count == 1
    with pytest.raises(GridFrozenError,
                       match=r"halo\['x'\]: demanded 5 > frozen 3"):
        grid.negotiate(halo=HaloSpec({"x": 5}))


# ================================================================
#  ConstantSpace-broadcast adoption
# ================================================================
def test_constant_broadcast_space_is_adopted(grid, mx, my):
    grid.freeze()
    profile = mx.constant * my.center
    assert profile not in grid.fingerprint.state_spaces
    report = grid.negotiate(state_spaces=(profile,))
    assert report.changed is False
    # adoption extends the record without reopening negotiation
    assert profile in grid.fingerprint.state_spaces


def test_adopted_space_passes_on_the_next_verify(grid, mx, my):
    grid.freeze()
    profile = mx.constant * my.center
    grid.negotiate(state_spaces=(profile,))
    report = grid.negotiate(state_spaces=(profile,))
    assert report.changed is False


def test_new_full_space_is_not_adopted(grid, mx, my):
    space = grid.create_field().function_space
    grid.negotiate(state_spaces=(space,), tendency=diff_x)
    grid.freeze()
    foreign = mx.right * my.center
    with pytest.raises(GridFrozenError,
                       match="not in the frozen record"):
        grid.negotiate(state_spaces=(foreign,), tendency=no_op)


# ================================================================
#  Post-freeze mutators
# ================================================================
def test_merge_overrides_raises_after_freeze(grid, mx):
    grid.freeze()
    with pytest.raises(GridFrozenError, match="frozen"):
        grid.merge_overrides(
            {("diff", mx.center): FiniteDifference(order=2)})


def test_with_immersed_raises_after_freeze(grid):
    grid.freeze()
    domain = ImmersedDomain(lambda x, y: 1.0)  # noqa: ARG005
    with pytest.raises(GridFrozenError, match="frozen"):
        grid.with_immersed(domain)


def test_grid_frozen_error_is_a_runtime_error():
    assert issubclass(GridFrozenError, RuntimeError)
