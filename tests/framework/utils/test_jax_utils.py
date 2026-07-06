"""Tests for the jax utilities."""
import weakref
from functools import partial
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr
from fridom.framework.timing_module import TimingComponent
from fridom.framework.utils import jax_utils
from fridom.framework.utils.jax_utils import (
    _AuxData,
    _eq_memo_lookup,
    _eq_memo_store,
    _function_eq,
    _object_eq,
    _structural_eq,
    _values_equal,
)


# ================================================================
#  jaxjit and free_memory
# ================================================================
def test_jaxjit():
    @fr.utils.jaxjit
    def square(x):
        return x**2

    assert square(3.0) == 9.0


def test_free_memory():
    x = jnp.arange(10.0)
    fr.utils.free_memory()
    assert x.is_deleted()
    # jax still works after freeing the buffers
    assert float(jnp.arange(10.0).sum()) == 45.0


# ================================================================
#  Structural value equality
# ================================================================
def test_arrays_equality():
    a = jnp.arange(4.0)
    assert _values_equal(a, jnp.arange(4.0))
    assert not _values_equal(a, jnp.arange(5.0))
    assert not _values_equal(a, jnp.zeros(4))
    # numpy and jax arrays are of different types
    assert not _values_equal(a, np.arange(4.0))


def test_number_equality_across_types():
    assert _values_equal(1, 1.0)
    assert not _values_equal(1, 2.0)
    assert not _values_equal(1, "1")


def test_container_equality():
    assert _values_equal([1, 2], [1, 2])
    assert not _values_equal([1, 2], [1, 3])
    assert _values_equal((1, jnp.ones(2)), (1, jnp.ones(2)))
    assert _values_equal({"a": 1}, {"a": 1})
    assert not _values_equal({"a": 1}, {"b": 1})


def test_function_equality():
    def make(power, offset=0.0):
        def func(x):
            return x**power + offset
        return func

    assert _values_equal(make(2), make(2))
    assert not _values_equal(make(2), make(3))

    def with_default(x, power=2):
        return x**power

    def with_other_default(x, power=3):
        return x**power

    assert not _function_eq(with_default, with_other_default)

    # same code object, different default values
    def make_with_default(default):
        def func(x, offset=default):
            return x + offset
        return func

    assert not _function_eq(make_with_default(1), make_with_default(2))


def test_function_equality_with_empty_cell():
    def make(define_value):
        def func():
            return value
        if define_value:
            value = None
        return func

    # the closure cells of the two functions are empty
    assert not _function_eq(make(define_value=False),
                            make(define_value=False))


def test_method_and_partial_equality():
    class MyClass:
        def method(self):
            return 1

    obj = MyClass()
    bound_a = MethodType(MyClass.method, obj)
    bound_b = MethodType(MyClass.method, obj)
    assert _values_equal(bound_a, bound_b)

    assert _values_equal(partial(sorted, reverse=True),
                         partial(sorted, reverse=True))
    assert not _values_equal(partial(sorted, reverse=True),
                             partial(sorted, reverse=False))


def test_structural_eq_of_jaxified_objects():
    diff_a = fr.grid.cartesian.FiniteDifferences()
    diff_b = fr.grid.cartesian.FiniteDifferences()
    assert diff_a == diff_b
    assert hash(diff_a) == hash(diff_b)

    # comparisons are memoized, so use a fresh object for the
    # inequality check
    diff_c = fr.grid.cartesian.FiniteDifferences()
    diff_c.custom_attribute = 42
    assert diff_a != diff_c

    assert _structural_eq(diff_a, "not a module") is NotImplemented


def test_object_eq_fast_paths():
    class Slotted:
        __slots__ = ("a",)

    class Dummy:
        pass

    obj = Dummy()
    assert _object_eq(obj, obj)
    assert not _object_eq(obj, 42)
    assert not _object_eq(Slotted(), Slotted())

    # objects with different attribute sets are not equal
    other = Dummy()
    other.extra = 1
    assert not _object_eq(obj, other)

    # the second comparison of the same pair is memoized
    a, b = Dummy(), Dummy()
    assert _object_eq(a, b)
    assert _object_eq(a, b)


def test_object_eq_with_reference_cycle():
    class Dummy:
        pass

    # the recursion through the attributes only happens for classes
    # from the fridom package
    Dummy.__module__ = "fridom.test_dummy"

    a, b = Dummy(), Dummy()
    a.ref = a
    b.ref = b
    assert _object_eq(a, b)


def test_values_equal_fridom_objects_without_custom_eq():
    assert _values_equal(TimingComponent("a"), TimingComponent("a"))
    assert not _values_equal(TimingComponent("a"), TimingComponent("b"))


def test_values_equal_with_raising_eq():
    class Weird:
        def __eq__(self, other):
            raise TypeError

        def __hash__(self):
            return 0

    assert not _values_equal(Weird(), Weird())


def test_structural_eq_identity():
    diff = fr.grid.cartesian.FiniteDifferences()
    assert _structural_eq(diff, diff)


# ================================================================
#  Comparison memoization
# ================================================================
def test_memo_store_and_lookup():
    class Dummy:
        pass

    a, b = Dummy(), Dummy()
    assert _eq_memo_lookup(a, b) is None

    _eq_memo_store(a, b, result=True)
    assert _eq_memo_lookup(a, b) is True


def test_memo_ignores_objects_without_weakref():
    # integers do not support weak references and are not memoized
    _eq_memo_store(1, 2, result=True)
    assert _eq_memo_lookup(1, 2) is None


def test_memo_detects_stale_entries():
    class Dummy:
        pass

    a, b = Dummy(), Dummy()
    other_a, other_b = Dummy(), Dummy()
    # fabricate a stale entry: the key matches, the references do not
    jax_utils._EQ_MEMO[(id(a), id(b))] = (
        weakref.ref(other_a), weakref.ref(other_b), True)

    assert _eq_memo_lookup(a, b) is None
    assert (id(a), id(b)) not in jax_utils._EQ_MEMO


def test_memo_is_cleared_when_full(monkeypatch):
    class Dummy:
        pass

    monkeypatch.setattr(jax_utils, "_EQ_MEMO_MAX_SIZE", 1)
    a, b = Dummy(), Dummy()
    c, d = Dummy(), Dummy()
    _eq_memo_store(a, b, result=True)
    _eq_memo_store(c, d, result=False)
    # the first entry was evicted when the memo was full
    assert _eq_memo_lookup(a, b) is None
    assert _eq_memo_lookup(c, d) is False


# ================================================================
#  AuxData
# ================================================================
def test_aux_data():
    aux_a = _AuxData({"a": 1, "b": jnp.ones(3)})
    aux_b = _AuxData({"a": 1, "b": jnp.ones(3)})
    aux_c = _AuxData({"a": 2, "b": jnp.ones(3)})

    assert aux_a == aux_b
    assert aux_a != aux_c
    assert hash(aux_a) == hash(aux_b)
    assert aux_a.__eq__("not aux data") is NotImplemented


# ================================================================
#  jaxify
# ================================================================
def test_jaxify_dynamic_must_be_tuple():
    with pytest.raises(TypeError):
        @partial(fr.utils.jaxify, dynamic="attr")
        class MyClass:
            pass


def test_jaxify_roundtrip():
    @partial(fr.utils.jaxify, dynamic=("arr",))
    class MyClass:
        def __init__(self, arr, power):
            self.arr = arr
            self.power = power

        @fr.utils.jaxjit
        def raise_to_power(self):
            return self.arr**self.power

    obj = MyClass(jnp.arange(3.0), 2)
    assert jnp.array_equal(obj.raise_to_power(), jnp.array([0.0, 1.0, 4.0]))

    # flatten and unflatten roundtrip
    children, aux_data = obj.tree_flatten()
    restored = MyClass.tree_unflatten(aux_data, children)
    assert jnp.array_equal(restored.arr, obj.arr)
    assert restored.power == obj.power


def test_jaxify_flatten_order_is_declaration_order():
    # flatten order must be deterministic (independent of
    # PYTHONHASHSEED) and follow the declaration order of the
    # dynamic attributes; a set-backed implementation breaks both
    @partial(fr.utils.jaxify, dynamic=("zulu", "alpha", "mike"))
    class Ordered:
        def __init__(self):
            self.zulu = jnp.array(0.0)
            self.alpha = jnp.array(1.0)
            self.mike = jnp.array(2.0)

    assert Ordered.dynamic_jax_attrs == ("zulu", "alpha", "mike")

    obj = Ordered()
    children, _ = obj.tree_flatten()
    assert [float(c) for c in children] == [0.0, 1.0, 2.0]

    leaves = jax.tree_util.tree_leaves(obj)
    assert [float(leaf) for leaf in leaves] == [0.0, 1.0, 2.0]

    # the roundtrip restores every attribute in place
    restored = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(obj), children)
    assert float(restored.zulu) == 0.0
    assert float(restored.alpha) == 1.0
    assert float(restored.mike) == 2.0


def test_jaxify_subclass_extends_dynamic_attrs_in_order():
    @partial(fr.utils.jaxify, dynamic=("whiskey", "delta"))
    class Base:
        pass

    # inherited attributes come first, then the newly declared ones
    @partial(fr.utils.jaxify, dynamic=("zeta", "beta"))
    class Sub(Base):
        pass

    assert Base.dynamic_jax_attrs == ("whiskey", "delta")
    assert Sub.dynamic_jax_attrs == ("whiskey", "delta", "zeta", "beta")

    # re-declaring an inherited attribute must not duplicate it
    @partial(fr.utils.jaxify, dynamic=("delta",))
    class SubDup(Base):
        pass

    assert SubDup.dynamic_jax_attrs == ("whiskey", "delta")


def test_jaxify_keeps_custom_hash():
    @fr.utils.jaxify
    class MyClass:
        def __hash__(self):
            return 42

    assert hash(MyClass()) == 42


def test_unflatten_without_dynamic_attrs():
    class Plain:
        pass

    obj = jax_utils._tree_unflatten.__func__(
        Plain, _AuxData({"a": 1}), ())
    assert obj.a == 1
    assert Plain.dynamic_jax_attrs == ()


# ================================================================
#  Inspection helper
# ================================================================
def test_inspect_jitted_function():
    @fr.utils.jaxjit
    def square(x):
        return x**2

    fr.utils.inspect_jitted_function(square, (jnp.arange(4.0),))
