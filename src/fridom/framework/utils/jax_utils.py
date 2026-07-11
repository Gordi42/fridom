"""Utilities for JAX operations."""
from __future__ import annotations

import types
import weakref
from functools import partial
from typing import Any, Generic, TypeVar

import jax
import numpy as np

import fridom.framework as fr

T = TypeVar("T")

def jaxjit(fun: callable, *args: Any, **kwargs: Any) -> callable:
    """
    Decorate a function for JAX JIT compilation.

    Description
    -----------
    This decorator is a thin wrapper around jax.jit. To disable jit
    compilation (e.g. for debugging), use the jax.disable_jit() context
    manager.

    Parameters
    ----------
    fun : callable
        The function to JIT compile.

    Returns
    -------
    `callable`
        The JIT compiled function.

    Examples
    --------
    >>> import fridom.framework as fr
    >>> @fr.utils.jaxjit
    ... def my_function(x):
    ...     return x**2
    """
    return jax.jit(fun, *args, **kwargs)

def free_memory() -> None:
    """
    Delete all live buffers in the JAX backend.

    Description
    -----------
    This function destroys all live buffers in the JAX backend. This is
    useful for rerunning the code in the same session without running out
    of memory.
    Note that the memory is only freed within JAX, not in the operating
    system. The operating system will still show the same memory usage.
    """
    for arr in jax.live_arrays():
        arr.delete()

# ================================================================
#  Structural equality (jit-cache friendly)
# ================================================================
# jax caches jit-compiled functions on the pytree structure of the
# arguments, which includes the auxiliary (static) data of jaxified
# objects. By default, python objects compare by identity, so two
# equivalent objects (e.g. two identical model settings created in two
# different tests) would never match and every jitted function would be
# recompiled. The helpers below implement a structural (by value)
# equality that is used both for the aux_data of jaxified objects and
# as the default `__eq__` of jaxified classes.

# pairs of object ids that are currently being compared; used to break
# reference cycles (e.g. mset <-> module back-references)
_EQ_IN_PROGRESS: set[tuple[int, int]] = set()

# memoized results of previous object comparisons; used to avoid
# repeated deep walks through large object graphs (e.g. on every jit
# dispatch). The weak references guard against id reuse after garbage
# collection.
_EQ_MEMO: dict[tuple[int, int],
               tuple[weakref.ref, weakref.ref, bool]] = {}
_EQ_MEMO_MAX_SIZE = 100_000


def _eq_memo_lookup(a: Any, b: Any) -> bool | None:
    """Look up a memoized comparison result (None if not memoized)."""
    entry = _EQ_MEMO.get((id(a), id(b)))
    if entry is None:
        return None
    ref_a, ref_b, result = entry
    if ref_a() is a and ref_b() is b:
        return result
    # stale entry (the ids were reused by new objects)
    del _EQ_MEMO[(id(a), id(b))]
    return None


def _eq_memo_store(a: Any, b: Any, result: bool) -> None:
    """Memoize the result of an object comparison."""
    try:
        entry = (weakref.ref(a), weakref.ref(b), result)
    except TypeError:
        # objects that do not support weak references are not memoized
        return
    if len(_EQ_MEMO) >= _EQ_MEMO_MAX_SIZE:
        _EQ_MEMO.clear()
    _EQ_MEMO[(id(a), id(b))] = entry


def _function_eq(a: types.FunctionType, b: types.FunctionType) -> bool:
    """Compare two functions by code, defaults and captured values."""
    if a.__code__ is not b.__code__:
        return False
    if not _values_equal(a.__defaults__, b.__defaults__):
        return False
    cells_a = a.__closure__ or ()
    cells_b = b.__closure__ or ()
    if len(cells_a) != len(cells_b):
        return False
    try:
        return all(
            _values_equal(ca.cell_contents, cb.cell_contents)
            for ca, cb in zip(cells_a, cells_b, strict=True))
    except ValueError:  # empty closure cell
        return False


def _object_eq(a: Any, b: Any) -> bool:
    """
    Compare two objects structurally by type and attributes.

    Description
    -----------
    Attributes listed in the class attribute `_eq_ignored_attrs` are
    excluded from the comparison. Classes use this to mark host-side
    state (e.g. progress bars, output writers, integration counters)
    that can never influence jit-compiled computations.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    dict_a = getattr(a, "__dict__", None)
    dict_b = getattr(b, "__dict__", None)
    if dict_a is None or dict_b is None:
        return False
    memoized = _eq_memo_lookup(a, b)
    if memoized is not None:
        return memoized
    key = (id(a), id(b))
    if key in _EQ_IN_PROGRESS:
        # cyclic reference; assume equality (the cycle entry point
        # will finish the comparison of all other attributes)
        return True
    _EQ_IN_PROGRESS.add(key)
    try:
        ignored = getattr(type(a), "_eq_ignored_attrs", frozenset())
        keys_a = set(dict_a) - ignored
        if keys_a != set(dict_b) - ignored:
            result = False
        else:
            result = all(
                _values_equal(dict_a[k], dict_b[k]) for k in keys_a)
    finally:
        _EQ_IN_PROGRESS.discard(key)
    _eq_memo_store(a, b, result)
    return result


def _values_equal(a: Any, b: Any) -> bool:  # noqa: PLR0911, C901
    """Compare two values structurally (array- and function-aware)."""
    if a is b:
        return True
    if isinstance(a, (np.ndarray, jax.Array)):
        if type(a) is not type(b):
            return False
        return (a.shape == b.shape
                and a.dtype == b.dtype
                and bool(np.array_equal(np.asarray(a), np.asarray(b))))
    if type(a) is not type(b):
        # values of different types are only considered equal when
        # both are plain python numbers
        if (isinstance(a, (bool, int, float, complex))
                and isinstance(b, (bool, int, float, complex))):
            return a == b
        return False
    if isinstance(a, (list, tuple)):
        return (len(a) == len(b)
                and all(map(_values_equal, a, b)))
    if isinstance(a, dict):
        return (a.keys() == b.keys()
                and all(_values_equal(v, b[k]) for k, v in a.items()))
    if isinstance(a, types.FunctionType):
        return _function_eq(a, b)
    if isinstance(a, types.MethodType):
        return (a.__func__ is b.__func__
                and _values_equal(a.__self__, b.__self__))
    if isinstance(a, partial):
        return (_values_equal(a.func, b.func)
                and _values_equal(a.args, b.args)
                and _values_equal(a.keywords, b.keywords))
    if (type(a).__eq__ is object.__eq__
            and type(a).__module__.partition(".")[0] == "fridom"):
        # fridom objects without custom equality: compare structurally
        # instead of by identity
        return _object_eq(a, b)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _structural_eq(self: Any, other: object) -> bool:
    """Default structural `__eq__` for jaxified classes."""
    if self is other:
        return True
    if type(self) is not type(other):
        return NotImplemented
    return _object_eq(self, other)


def _structural_hash(self: Any) -> int:
    """Default `__hash__` for jaxified classes (consistent with eq)."""
    return hash(type(self))


class _AuxData:

    """
    Hashable wrapper around the aux data of jaxified objects.

    Description
    -----------
    The auxiliary (static) data of a jaxified object is part of the
    pytree structure and therefore part of the jit-cache key. This
    wrapper provides a tolerant, structural equality so that equivalent
    objects (e.g. from two identical model setups) hit the same jit
    cache entry instead of triggering a recompilation.

    Attributes named in `annotations` (the class's opt-in
    `annotation` category, see `jaxify`) are carried in the aux data
    — they survive flatten/unflatten — but are *excluded from the
    equality*: two objects differing only in annotation attributes
    produce equal pytree structures (no retrace, scan-carry stable).
    """

    __slots__ = ("annotations", "data")

    def __init__(self, data: dict,
                 annotations: frozenset = frozenset()) -> None:
        self.data = data
        self.annotations = annotations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _AuxData):
            return NotImplemented
        if self.annotations != other.annotations:
            return False
        if not self.annotations:
            return _values_equal(self.data, other.data)
        if self.data.keys() != other.data.keys():
            return False
        return all(
            _values_equal(value, other.data[key])
            for key, value in self.data.items()
            if key not in self.annotations)

    def __hash__(self) -> int:
        return hash(frozenset(self.data.keys()))


def _merge_dynamic_attrs(
        cls: type, dynamic: tuple[str] | None) -> tuple[str, ...]:
    """Validate `dynamic` and merge it with inherited dynamic attributes.

    The merged attributes preserve declaration order (inherited
    attributes first, then the newly declared ones): the pytree
    flatten order of jaxified objects must be deterministic across
    processes, so an unordered container (whose iteration order
    depends on ``PYTHONHASHSEED``) must never be used here.
    """
    # make sure dynamic is either a tuple or None:
    if not isinstance(dynamic, (tuple, type(None))):
        fr.log.error("dynamic must be a tuple or None, not %s", type(dynamic))
        fr.log.error("In case you only have one dynamic attribute, ")
        fr.log.error("use dynamic=('attr',) instead of dynamic=('attr').")
        raise TypeError

    # inherited dynamic attributes first (they were declared first),
    # then the new ones, removing duplicates while keeping order
    merged = list(getattr(cls, "dynamic_jax_attrs", ()))
    merged += [attr for attr in (dynamic or ()) if attr not in merged]
    return tuple(merged)

def _merge_annotation_attrs(
        cls: type, annotation: tuple[str] | None) -> frozenset:
    """Validate `annotation` and merge it with inherited ones.

    Annotation attributes are excluded from the aux-data equality
    (not from the aux data itself), so no ordering is required; the
    merged set is a plain frozenset of attribute names.
    """
    # make sure annotation is either a tuple or None:
    if not isinstance(annotation, (tuple, type(None))):
        fr.log.error(
            "annotation must be a tuple or None, not %s",
            type(annotation))
        fr.log.error("In case you only have one annotation attribute,")
        fr.log.error("use annotation=('attr',) instead of "
                     "annotation=('attr').")
        raise TypeError

    inherited = getattr(cls, "annotation_jax_attrs", frozenset())
    return frozenset(inherited) | frozenset(annotation or ())

def _tree_flatten(self: T) -> tuple[tuple, _AuxData]:
    """Flatten a jaxified object into (children, aux_data)."""
    # Store all attributes that are marked as dynamic
    children = tuple(
        getattr(self, attr) for attr in self.dynamic_jax_attrs)

    # Store all other attributes as aux_data; attributes in the
    # annotation category are carried but exempt from aux equality
    aux_data = _AuxData({key: att for key, att in self.__dict__.items()
                         if key not in self.dynamic_jax_attrs},
                        annotations=self.annotation_jax_attrs)

    return (children, aux_data)

@classmethod
def _tree_unflatten(cls: type[T], aux_data: _AuxData, children: tuple) -> T:
    """Reconstruct a jaxified object from (aux_data, children)."""
    obj = object.__new__(cls)
    # be paranoid and check that the class has the
    # dynamic_jax_attrs attribute
    if not hasattr(cls, "dynamic_jax_attrs"):
        # this should never happen
        fr.log.error(
            "The class %s does not have the dynamic_jax_attrs "
            "attribute.", cls)
        cls.dynamic_jax_attrs = ()
    # set static attributes
    for key, value in aux_data.data.items():
        setattr(obj, key, value)
    # set dynamic attributes
    for i, attr in enumerate(cls.dynamic_jax_attrs):
        setattr(obj, attr, children[i])
    return obj

def jaxify(cls: Generic[T], dynamic: tuple[str] | None = None,
           annotation: tuple[str] | None = None) -> T:
    """
    Add JAX pytree support to a class (for jit compilation).

    Description
    -----------
    In order to use jax.jit on custom classes, the class must be registered
    to jax. This decorator adds the necessary methods to the class to make it
    compatible with jax.jit.
    By default, all attributes of an object are considered static, i.e., they
    they will not be traced by jax. Attributes that should be dynamic must
    be marked specified with the `dynamic` argument.
    Static attributes named in the `annotation` argument form the
    annotation category: they stay in the static aux data (they
    survive flatten/unflatten) but are excluded from the aux-data
    equality, so two objects differing only in annotation attributes
    have equal pytree structures — jit caching, `lax.scan` carries,
    and `vmap` are insensitive to them, and objects returned from
    jitted functions carry trace-time annotation values. This is
    opt-in per class; classes that do not declare it are unaffected.
    Subclasses of a jaxified class are automatically registered as pytrees
    as well; they only need to apply this decorator themselves when they
    want to mark additional attributes as dynamic.

    .. note::
        The `dynamic` argument must be a tuple of attribute names. If
        you only have one dynamic attribute, use dynamic=('attr',)
        instead of dynamic=('attr').

    .. note::
        If a static attribute is changed, all jit compiled functions of
        the class must be recompiled. Hence, such attributes should be
        marked as dynamic. However, marking an attribute as dynamic will
        increase the computational cost. So, it is advisable to only
        mark attributes as dynamic that are actually changing during the
        simulation.

    .. warning::
        Methods that are jit compiled with fr.utils.jaxjit will not modify the
        object in place.

    Parameters
    ----------
    cls : type
        The class to add jax support to.
    dynamic : tuple[str] | None, optional
        A tuple of attribute names that should be considered dynamic (default:
        None).
    annotation : tuple[str] | None, optional
        A tuple of static attribute names that are exempt from the
        aux-data equality (pure annotation, e.g. field metadata;
        default: None).

    Examples
    --------
    A class with no dynamic attributes:

    .. code-block:: python

        import fridom.framework as fr

        @fr.utils.jaxify
        class MyClass:
            _dynamic_attributes = ["x",]
            def __init__(self, power):
                self.power = power

            @fr.utils.jaxjit
            def raise_to_power(self, arr):
                return arr**self.power

    A class with dynamic attributes:

    .. code-block:: python

        import fridom.framework as fr
        from functools import partial

        @partial(fr.utils.jaxify, dynamic=('arr',))
        class MyClass:
            def __init__(self, arr, power):
                self.power = power
                self.arr = arr

            @fr.utils.jaxjit
            def raise_to_power(self):
                return self.arr**self.power
    """
    # set the merged dynamic and annotation attributes on the class
    cls.dynamic_jax_attrs = _merge_dynamic_attrs(cls, dynamic)
    cls.annotation_jax_attrs = _merge_annotation_attrs(cls, annotation)

    # set the flatten/unflatten methods on the class
    cls.tree_unflatten = _tree_unflatten
    cls.tree_flatten = _tree_flatten

    # give the class a structural (by value) equality so that
    # equivalent objects produce equal pytree structures and hence hit
    # the same jit-cache entries (classes with a custom __eq__ are left
    # untouched)
    if cls.__eq__ is object.__eq__:
        cls.__eq__ = _structural_eq
        if cls.__hash__ is object.__hash__:
            cls.__hash__ = _structural_hash

    # register the class with jax
    # a class may reach this point twice, for example when a subclass of a
    # jaxified class (automatically registered on creation, see below) is
    # decorated with @jaxify to mark additional dynamic attributes. In that
    # case we only update `dynamic_jax_attrs` (done above) and skip the
    # registration.
    if not cls.__dict__.get("_jaxify_registered", False):
        jax.tree_util.register_pytree_node(
            cls, cls.tree_flatten, cls.tree_unflatten)
        cls._jaxify_registered = True

    # automatically register subclasses as pytrees
    if not cls.__dict__.get("_jaxify_hooked", False):
        def _auto_jaxify(sub_cls: type, **kwargs: dict) -> None:
            super(cls, sub_cls).__init_subclass__(**kwargs)
            jaxify(sub_cls)
        cls.__init_subclass__ = classmethod(_auto_jaxify)
        cls._jaxify_hooked = True

    return cls

# ================================================================
#  Inspect jax jit functions
# ================================================================
def inspect_jitted_function(func: callable, args: tuple) -> None:
    """
    Inspect if a jit compiled function has communication operations.

    Parameters
    ----------
    func : callable
        The function to inspect.
    args : tuple
        The arguments to pass to the function. Must be a tuple.
    """
    func.lower(*args).compile().runtime_executable().hlo_modules()[0].to_string()
    patterns = ["all-gather",
                "all-reduce",
                "all-to-all",
                "scatter",
                "gather",
                "cross-replica-sum",
                "collective-permute",
                "dynamic-slice"]
    for _pattern in patterns:
        pass
