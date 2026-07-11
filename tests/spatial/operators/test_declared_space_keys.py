"""Tests for the mesh-keyed ``("declared_space", mesh)`` resolver rows."""
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.operators.registry import (
    DispatchError,
    OperatorRegistry,
)


@pytest.fixture
def resolver(mx):
    def resolve(tag, bc):  # noqa: ARG001 — the D1.2 signature
        return mx.center

    return resolve


# ================================================================
#  Registration and resolution (exact mapping surface)
# ================================================================
def test_constructor_defaults_seed_resolver_rows(mx, resolver):
    reg = OperatorRegistry({("declared_space", mx): resolver})
    assert ("declared_space", mx) in reg
    assert reg[("declared_space", mx)] is resolver


def test_setitem_registers_a_resolver_row(mx, resolver):
    reg = OperatorRegistry()
    reg[("declared_space", mx)] = resolver
    assert ("declared_space", mx) in reg
    assert reg[("declared_space", mx)] is resolver


def test_resolver_row_resolves_a_declaration(mx, resolver):
    reg = OperatorRegistry({("declared_space", mx): resolver})
    space = reg[("declared_space", mx)]("collocated", BC.NONE)
    assert space is mx.center


def test_missing_resolver_is_a_dispatch_error(mx):
    reg = OperatorRegistry()
    assert ("declared_space", mx) not in reg
    with pytest.raises(DispatchError, match="no resolver"):
        reg[("declared_space", mx)]


def test_resolver_values_must_be_callable(mx):
    reg = OperatorRegistry()
    with pytest.raises(TypeError, match="callable"):
        reg[("declared_space", mx)] = 42


def test_resolver_rows_are_absent_from_items(mx, resolver, keep_cls):
    k = keep_cls()
    reg = OperatorRegistry({("declared_space", mx): resolver,
                            ("keep", mx.center): k})
    assert dict(reg.items()) == {("keep", mx.center): k}


def test_resolver_rows_never_enter_operator_dispatch(mx, resolver):
    reg = OperatorRegistry({("declared_space", mx): resolver})
    # resolve() is the operator surface; resolver rows are not
    # operators and must not leak into it
    with pytest.raises(DispatchError, match="no operator"):
        reg.resolve("declared_space", mx.center)


# ================================================================
#  Never module-mergeable (model D1.2)
# ================================================================
def test_merge_carries_the_resolver_table_forward(mx, resolver,
                                                  keep_cls):
    reg = OperatorRegistry({("declared_space", mx): resolver})
    merged = reg.merge({"keep": keep_cls()})
    assert merged[("declared_space", mx)] is resolver


def test_merged_resolver_table_is_isolated(mx, my, resolver):
    reg = OperatorRegistry({("declared_space", mx): resolver})
    merged = reg.merge({})
    merged[("declared_space", my)] = resolver
    assert ("declared_space", my) not in reg


def test_merge_rejects_mesh_keyed_overrides(mx, resolver):
    reg = OperatorRegistry()
    with pytest.raises(ValueError, match="never module-mergeable"):
        reg.merge({("declared_space", mx): resolver})


def test_merge_rejects_space_keyed_declared_space(mx, keep_cls):
    reg = OperatorRegistry()
    with pytest.raises(ValueError, match="never module-mergeable"):
        reg.merge({("declared_space", mx.center): keep_cls()})


def test_merge_rejects_kind_only_declared_space(keep_cls):
    reg = OperatorRegistry()
    with pytest.raises(ValueError, match="never module-mergeable"):
        reg.merge({"declared_space": keep_cls()})


def test_facade_rejects_declared_space_overrides(mx, resolver):
    grid = Grid((mx,))
    with pytest.raises(ValueError, match="never module-mergeable"):
        grid.merge_overrides({("declared_space", mx): resolver})
    with pytest.raises(ValueError, match="never module-mergeable"):
        grid.merge_overrides(
            {"mod": {("declared_space", mx): resolver}})


def test_grid_level_registration_stays_open(mx, resolver):
    # the grid builder seeds/overrides resolver rows directly on the
    # registry (grid-level, "once by the grid builder" — D1.2)
    grid = Grid((mx,))
    grid.dispatch[("declared_space", mx)] = resolver
    assert grid.dispatch[("declared_space", mx)] is resolver
