"""Tests for the ``Grid.merge_overrides`` facade (grid.md)."""
import pytest

from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import GridFrozenError
from fridom.spatial.grid import Grid
from fridom.spatial.operators.registry import (
    DispatchCollisionError,
    DispatchError,
    OperatorRegistry,
)


@pytest.fixture
def grid(mx):
    return Grid((mx,))


# ================================================================
#  Facade: merge + swap of the held registry instance
# ================================================================
def test_facade_merges_and_swaps_the_registry(grid, mx, keep_cls):
    old = grid.dispatch
    k = keep_cls()
    grid.merge_overrides({"keep": k})
    assert isinstance(grid.dispatch, OperatorRegistry)
    assert grid.dispatch is not old
    assert grid.dispatch.resolve("keep", mx.center) is k
    # the pure registry merge never mutates the old instance
    with pytest.raises(DispatchError, match="no operator"):
        old.resolve("keep", mx.center)


def test_space_keyed_override_shadows_the_default(grid, mx,
                                                  stagger_cls):
    mine = stagger_cls()
    default = grid.dispatch.resolve("diff", mx.right)
    grid.merge_overrides({("diff", mx.center): mine})
    assert grid.dispatch.resolve("diff", mx.center) is mine
    # unoverridden defaults stay visible through the merged layers
    assert grid.dispatch.resolve("diff", mx.right) is default


def test_per_module_form_merges_all_modules(grid, mx, keep_cls,
                                            stagger_cls):
    k, s = keep_cls(), stagger_cls()
    grid.merge_overrides({
        "modA": {"keep": k},
        "modB": {("diff", mx.center): s},
    })
    assert grid.dispatch.resolve("keep", mx.center) is k
    assert grid.dispatch.resolve("diff", mx.center) is s


# ================================================================
#  Duplicate resolved key across two modules
# ================================================================
def test_duplicate_key_across_modules_names_both(grid, mx,
                                                 stagger_cls):
    with pytest.raises(DispatchCollisionError,
                       match=r"modA.*modB"):
        grid.merge_overrides({
            "modA": {("diff", mx.center): stagger_cls()},
            "modB": {("diff", mx.center): stagger_cls()},
        })


def test_duplicate_detection_normalizes_keys(grid, mx, stagger_cls):
    # a laid-out and a bare spelling of one space are the same
    # resolved key (dispatch keys never see layouts)
    laid = mx.center.with_layout(Layout({}))
    with pytest.raises(DispatchCollisionError,
                       match=r"modA.*modB"):
        grid.merge_overrides({
            "modA": {("diff", laid): stagger_cls()},
            "modB": {("diff", mx.center): stagger_cls()},
        })


def test_colliding_merge_leaves_the_registry_untouched(grid, mx,
                                                       stagger_cls):
    old = grid.dispatch
    with pytest.raises(DispatchCollisionError, match="modA"):
        grid.merge_overrides({
            "modA": {("diff", mx.center): stagger_cls()},
            "modB": {("diff", mx.center): stagger_cls()},
        })
    assert grid.dispatch is old


def test_distinct_keys_across_modules_do_not_collide(grid, mx,
                                                     stagger_cls):
    a, b = stagger_cls(), stagger_cls()
    grid.merge_overrides({
        "modA": {("diff", mx.center): a},
        "modB": {("diff", mx.right): b},
    })
    assert grid.dispatch.resolve("diff", mx.center) is a
    assert grid.dispatch.resolve("diff", mx.right) is b


# ================================================================
#  Pre-freeze guard
# ================================================================
def test_merge_after_freeze_raises_grid_frozen_error(grid, keep_cls):
    grid.freeze()
    with pytest.raises(GridFrozenError, match="frozen"):
        grid.merge_overrides({"keep": keep_cls()})


def test_even_empty_overrides_raise_after_freeze(grid):
    grid.freeze()
    with pytest.raises(GridFrozenError, match="assembly phase"):
        grid.merge_overrides({})


def test_grid_frozen_error_is_a_runtime_error():
    # grid.md: post-freeze merges "raise RuntimeError"
    assert issubclass(GridFrozenError, RuntimeError)
