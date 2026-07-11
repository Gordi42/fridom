"""Tests for freeze(): fingerprint, verify path, GridFrozenError."""
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
def test_verify_fails_on_larger_extra_halo(grid):
    grid.freeze()
    frozen_x = grid.fingerprint.halo["x"]
    with pytest.raises(GridFrozenError,
                       match=rf"halo\['x'\]: demanded "
                             rf"{frozen_x + 2} > frozen {frozen_x}"):
        grid.negotiate(halo=HaloSpec({"x": frozen_x + 2}))


def test_verify_merges_extra_halo_with_the_trace(grid):
    # merge_max rule: halo= widens the traced demand, it never
    # overrides it — the extra spec alone must be able to fail
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
