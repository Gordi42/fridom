"""array_ops.py - Utilities for array operations."""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

import numpy as np

import fridom.framework as fr

T = TypeVar("T")

class SliceableAttribute(Generic[T]):

    """
    Class to make an object sliceable.

    Parameters
    ----------
    slicer : Callable
        The slicer function.

    """

    def __init__(self, slicer: Callable[[int | slice | tuple[int, slice]], T]) -> None:
        self.slicer = slicer

    def __getitem__(self, key: int | slice | tuple[int, slice]) -> T:
        return self.slicer(key)

def modify_array(arr: np.ndarray, where: slice, value: np.ndarray) -> np.ndarray:
    """
    Return a new array with the modifications.
    
    Description
    -----------
    A fundamental difference between JAX and NumPy is that NumPy allows
    in-place modification of arrays, while JAX does not. This function does 
    not modify the input array in place, but returns a new array with the
    modifications.
    
    Parameters
    ----------
    `arr` : `np.ndarray`
        The array to modify.
    `where` : `slice`
        The slice to modify.
    `value` : `np.ndarray | float | int`
        The value to set.
    
    Returns
    -------
    `np.ndarray`
        The modified array.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> x = fr.config.ncp.arange(10)  # create some array
    >>> # instead of x[2:5] = 0, we use the modify_array function
    >>> x = fr.utils.modify_array(x, slice(2,5), 0)

    """
    if fr.config.backend_is_jax:
        return arr.at[where].set(value)
    res = arr.copy()
    res[where] = value
    return res

def random_array(shape: tuple[int], seed=12345, **kwargs) -> np.ndarray:
    """Create a random array."""
    if "ignore_warning" not in kwargs:
        fr.log.warning("The random_array function is deprecated and will be removed in the future.")
        fr.log.warning("Please use the create array method from the grid object instead")
    if fr.config.backend_is_jax:
        # we need to import jax here since it is an optional dependency
        import jax  # pylint: disable=import-outside-toplevel
        key = jax.random.key(seed)
        return jax.random.normal(key, shape)
    ncp = fr.config.ncp
    default_rng = ncp.random.default_rng
    return default_rng(seed).standard_normal(shape)

def array_is_constant(arr: np.ndarray) -> bool:
    """
    Check if an array is constant.

    Description
    -----------
    This function checks if all elements of an array are the same.

    Parameters
    ----------
    arr : np.ndarray
        The array to check.

    Returns
    -------
    bool
        True if the array is constant, False otherwise.

    Examples
    --------
    >>> import fridom.framework as fr
    >>> x = fr.config.ncp.ones(10)
    >>> fr.utils.array_is_constant(x)
    True
    >>> x[5] = 0
    >>> fr.utils.array_is_constant(x)
    False

    """
    ncp = fr.config.ncp
    return ncp.allclose(arr, ncp.mean(arr))
