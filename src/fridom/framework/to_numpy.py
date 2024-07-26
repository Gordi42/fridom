# Import external modules
from copy import deepcopy
from mpi4py import MPI
import numpy as np
import inspect
class No_Type: ...
try:
    import cupy as cp
    cupy_array = cp.ndarray
except ImportError:
    cp = None
    cupy_array = No_Type
try:
    import jax
    jax_array = jax.numpy.ndarray
except ImportError:
    jax = None
    jax_array = No_Type
# Import internal modules
from fridom.framework import config


def to_numpy(obj, memo=None, _nil=[]):
    # if the backend is numpy, return a deepcopy
    if config.backend == 'numpy':
        return deepcopy(obj)

    # if the object has a _cpu attribute which is not None, return it
    if hasattr(obj, '_cpu'):
        if obj._cpu is not None:
            return obj._cpu

    # if the object was already converted to numpy, return it (recursive call)
    if memo is None:
        memo = {}

    d = id(obj)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y
    
    # if the object has a __to_numpy__ method, call it
    if hasattr(obj, '__to_numpy__'):
        memo[d] = obj.__to_numpy__(memo)

    # if the object is a cupy array, convert it to numpy and return it
    elif isinstance(obj, cupy_array):
        memo[d] = cp.asnumpy(obj)
    
    # if the object is a jax array, convert it to numpy and return it
    elif isinstance(obj, jax_array):
        memo[d] = np.array(obj)

    # if the object is a numpy array, return it
    elif isinstance(obj, (np.ndarray, np.generic)):
        memo[d] = deepcopy(obj)

    # if the object is a module, return it
    elif inspect.ismodule(obj):
        memo[d] = obj

    # if the object is a function, return it
    elif inspect.isfunction(obj):
        memo[d] = obj

    # if the object is a method, return it
    elif inspect.ismethod(obj):
        memo[d] = obj

    # if the object is a dictionary, convert all values
    elif isinstance(obj, dict):
        memo[d] = {key: to_numpy(value, memo) for key, value in obj.items()}

    # if the object is a list, convert all elements
    elif isinstance(obj, list):
        memo[d] = [to_numpy(x, memo) for x in obj]

    # if the object is a tuple, convert all elements
    elif isinstance(obj, tuple):
        memo[d] = tuple(to_numpy(x, memo) for x in obj)

    # if the object is a set, convert all elements
    elif isinstance(obj, set):
        memo[d] = {to_numpy(x, memo) for x in obj}

    # if the object is a type, return it
    elif isinstance(obj, type):
        memo[d] = deepcopy(obj)

    # if the object is a MPI.Cartcomm, return it
    elif isinstance(obj, MPI.Cartcomm):
        memo[d] = obj

    # if the object is not a python object, return a deepcopy
    elif not hasattr(obj, '__dict__'):
        memo[d] = deepcopy(obj)
    
    # if the object is a python object, convert all attributes
    else:
        memo[d] = deepcopy(obj)
        for key, value in vars(obj).items():
            setattr(memo[d], key, to_numpy(value, memo))

    if hasattr(obj, '_cpu'):
        obj._cpu = memo[d]

    return memo[d]

    
