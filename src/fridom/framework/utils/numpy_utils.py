"""numpy_utils.py - Utilities for numpy operations."""
from copy import deepcopy
import inspect
from typing import Union
import numpy as np
import fridom.framework as fr
if fr.utils.MPI_AVAILABLE:
    from mpi4py import MPI
else:
    MPI = None


def _handle_to_numpy(obj: object, memo: dict) -> object:
    """Handle objects with a __to_numpy__ method."""
    return obj.__to_numpy__(memo)

def _handle_cpu(obj: object) -> object:
    """Handle objects with a _cpu attribute."""
    return obj._cpu  # pylint: disable=protected-access

def _handle_ndarray(obj: np.ndarray) -> np.ndarray:
    """Handle ndarrays based on the backend."""
    match fr.config.backend:
        case "numpy":
            return deepcopy(obj)
        case "cupy":
            return fr.config.ncp.asnumpy(obj)
        case "jax_cpu" | "jax_gpu":
            return np.array(obj)

def _handle_iterable(obj: Union[dict, list, tuple, set],
                     memo: dict) -> Union[dict, list, tuple, set]:
    """Handle dictionaries, lists, tuples, and sets."""
    if isinstance(obj, dict):
        return {key: to_numpy(value, memo) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_numpy(x, memo) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_numpy(x, memo) for x in obj)
    if isinstance(obj, set):
        return {to_numpy(x, memo) for x in obj}
    # if none of the above, raise an error
    raise TypeError(f"Object of type {type(obj)} is not iterable.")

def _handle_python_object(obj: object, memo: dict) -> object:
    """Handle generic Python objects with attributes."""
    d = id(obj)
    memo[d] = deepcopy(obj)
    for key, value in vars(obj).items():
        setattr(memo[d], key, to_numpy(value, memo))
    return memo[d]

def _create_numpy_copy(obj: object, memo: dict) -> object:
    """Create a numpy-compatible copy of the object."""
    if hasattr(obj, '__to_numpy__'):
        result = _handle_to_numpy(obj, memo)

    elif hasattr(obj, '_cpu') and obj._cpu is not None:  # pylint: disable=protected-access
        result = _handle_cpu(obj)

    elif isinstance(obj, fr.config.ncp.ndarray):
        result = _handle_ndarray(obj)

    elif isinstance(obj, (np.ndarray, np.generic)):
        result = deepcopy(obj)

    elif inspect.ismodule(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
        result = obj

    elif isinstance(obj, (dict, list, tuple, set)):
        result = _handle_iterable(obj, memo)

    elif isinstance(obj, type):
        result = deepcopy(obj)

    elif fr.utils.MPI_AVAILABLE and isinstance(obj, MPI.Cartcomm):
        result = obj

    elif not hasattr(obj, '__dict__'):
        result = deepcopy(obj)

    else:
        result = _handle_python_object(obj, memo)

    return result

def to_numpy(obj: object, memo: dict | None = None, _nil: list = None) -> object:
    """
    Creates a deep copy of an object with all arrays converted to numpy.
    
    Description
    -----------
    Some functions require numpy arrays as input, as for example plotting
    with matplotlib. This function creates a deep copy of an object where
    all arrays are converted to numpy arrays. This is computationally
    expensive and should be used with care. Objects that should only be
    converted once, as for example the grid variables which are usually
    static, i.e. they do not change during the simulation, should have a
    _cpu attribute. If the _cpu attribute is None, the object is converted
    to numpy and cached in the _cpu attribute. If the _cpu attribute is not
    None, the cached numpy array is returned. Objects that require a 
    custom conversion should implement a __to_numpy__ method that returns
    the converted object.
    
    Parameters
    ----------
    `obj` : `Any`
        The object to convert to numpy.
    `memo` : `dict` (default=None)
        A dictionary to store the converted objects (used for recursion).
    
    Returns
    -------
    `Any`
        The object with all arrays converted to numpy.
    """
    _nil = _nil or []
    # if the backend is numpy, return a deepcopy
    if fr.config.backend == 'numpy':
        return deepcopy(obj)

    # if the object was already converted to numpy, return it (recursive call)
    if memo is None:
        memo = {}

    d = id(obj)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    memo[d] = _create_numpy_copy(obj, memo)

    if hasattr(obj, '_cpu'):
        obj._cpu = memo[d]  # pylint: disable=protected-access

    return memo[d]

def to_seconds(t: Union[float, np.datetime64, np.timedelta64]) -> float:
    """
    Convert a time to seconds.
    
    Description
    -----------
    This function converts a time to seconds. The time can be given as a
    float, a np.datetime64 or a np.timedelta64.
    
    Parameters
    ----------
    `t` : `Union[float, np.datetime64, np.timedelta64]`
        The time to convert to seconds.
    
    Returns
    -------
    `float`
        The time in seconds.
    """
    if isinstance(t, np.timedelta64):
        # Conversion factors for common time units to seconds
        conversion_factors = {
            'Y': 365 * 24 * 3600,       # 365 days
            'M': 30 * 24 * 3600,        # 30 days
            'W': 7 * 24 * 3600,         # 7 days
            'D': 24 * 3600,             # 1 day
            'h': 3600,                  # 1 hour
            'm': 60,                    # 1 minute
            's': 1                      # 1 second
        }

        # Get the time unit of the timedelta64 object (e.g., 'Y', 'M', 'D')
        unit = np.datetime_data(t)[0]

        # Calculate the seconds based on the conversion factor
        return t / np.timedelta64(1, unit) * conversion_factors[unit]

    if isinstance(t, np.datetime64):
        t = t.astype('datetime64[s]')  # convert time stemp to seconds
        return float(t.astype('timedelta64[s]').astype(float))
    return t
