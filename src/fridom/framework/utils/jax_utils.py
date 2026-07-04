"""Utilities for JAX operations."""
from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax

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
    backend = jax.extend.backend.get_backend()
    for buf in backend.live_buffers():
        buf.delete()

def _merge_dynamic_attrs(
        cls: type, dynamic: tuple[str] | None) -> set[str]:
    """Validate `dynamic` and merge it with inherited dynamic attributes."""
    # make sure dynamic is either a tuple or None:
    if not isinstance(dynamic, (tuple, type(None))):
        fr.log.error("dynamic must be a tuple or None, not %s", type(dynamic))
        fr.log.error("In case you only have one dynamic attribute, ")
        fr.log.error("use dynamic=('attr',) instead of dynamic=('attr').")
        raise TypeError

    dynamic = list(dynamic or [])

    # merge with the (possibly inherited) dynamic attributes
    if hasattr(cls, "dynamic_jax_attrs"):
        dynamic += list(cls.dynamic_jax_attrs)

    # remove duplicates
    return set(dynamic)

def _tree_flatten(self: T) -> tuple[tuple, dict]:
    """Flatten a jaxified object into (children, aux_data)."""
    # Store all attributes that are marked as dynamic
    children = tuple(
        getattr(self, attr) for attr in self.dynamic_jax_attrs)

    # Store all other attributes as aux_data
    aux_data = {key: att for key, att in self.__dict__.items()
                if key not in self.dynamic_jax_attrs}

    return (children, aux_data)

@classmethod
def _tree_unflatten(cls: type[T], aux_data: dict, children: tuple) -> T:
    """Reconstruct a jaxified object from (aux_data, children)."""
    obj = object.__new__(cls)
    # be paranoid and check that the class has the
    # dynamic_jax_attrs attribute
    if not hasattr(cls, "dynamic_jax_attrs"):
        # this should never happen
        fr.log.error(
            "The class %s does not have the dynamic_jax_attrs "
            "attribute.", cls)
        cls.dynamic_jax_attrs = set()
    # set static attributes
    for key, value in aux_data.items():
        setattr(obj, key, value)
    # set dynamic attributes
    for i, attr in enumerate(cls.dynamic_jax_attrs):
        setattr(obj, attr, children[i])
    return obj

def jaxify(cls: Generic[T], dynamic: tuple[str] | None = None) -> T:
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
    # set the merged dynamic attributes on the class
    cls.dynamic_jax_attrs = _merge_dynamic_attrs(cls, dynamic)

    # set the flatten/unflatten methods on the class
    cls.tree_unflatten = _tree_unflatten
    cls.tree_flatten = _tree_flatten

    # register the class with jax
    jax.tree_util.register_pytree_node(
        cls, cls.tree_flatten, cls.tree_unflatten)

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
