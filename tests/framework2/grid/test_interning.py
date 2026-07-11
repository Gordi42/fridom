"""Tests for the weak intern table (framework2/grid/interning.py)."""
import gc

from fridom.spatial.interning import InternTable


class Interned:

    """Weak-referenceable stand-in for an interned space/product."""

    def __init__(self, payload=None):
        self.payload = payload


# ================================================================
#  Identity guarantee
# ================================================================
def test_same_key_returns_the_identical_object():
    table = InternTable()
    a = table.intern(("Center", "REAL"), Interned)
    b = table.intern(("Center", "REAL"), Interned)
    assert a is b


def test_different_keys_return_different_objects():
    table = InternTable()
    a = table.intern(("Center", "REAL"), Interned)
    b = table.intern(("Center", "COMPLEX"), Interned)
    assert a is not b
    assert len(table) == 2


def test_factory_runs_only_on_first_request():
    table = InternTable()
    calls = []

    def factory():
        calls.append(1)
        return Interned()

    first = table.intern("key", factory)
    second = table.intern("key", factory)
    assert first is second
    assert len(calls) == 1


def test_value_equal_keys_are_the_same_entry():
    table = InternTable()
    a = table.intern(("Center", ("x",), 8), Interned)
    b = table.intern(("Center", ("x",), 8), Interned)
    assert a is b
    assert "key" not in table
    assert ("Center", ("x",), 8) in table


# ================================================================
#  Collectability (weak values)
# ================================================================
def test_unreferenced_entries_are_collected():
    table = InternTable()
    obj = table.intern("key", Interned)
    assert len(table) == 1

    del obj
    gc.collect()
    assert len(table) == 0
    assert "key" not in table


def test_collected_entries_are_rebuilt_on_demand():
    table = InternTable()
    first_id = id(table.intern("key", Interned))
    gc.collect()

    # the entry was collected, so the factory runs again
    calls = []

    def factory():
        calls.append(1)
        return Interned()

    rebuilt = table.intern("key", factory)
    assert len(calls) == 1
    assert isinstance(rebuilt, Interned)
    # freed identity may or may not be recycled; only liveness matters
    assert first_id is not None


def test_live_entries_are_not_collected():
    table = InternTable()
    keep = table.intern("key", Interned)
    gc.collect()
    assert len(table) == 1
    assert table.intern("key", Interned) is keep


# ================================================================
#  No cross-table / cross-test leakage
# ================================================================
def test_tables_are_independent():
    table_a = InternTable()
    table_b = InternTable()
    a = table_a.intern("key", Interned)
    b = table_b.intern("key", Interned)
    assert a is not b


def test_fresh_table_starts_empty():
    # guards against module-level state leaking between tests
    assert len(InternTable()) == 0
