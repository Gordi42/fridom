"""
Utility functions and classes for the FRIDOM framework.
"""
from typing import Union, TypeVar, Generic
from . import config
from .config import logger
from mpi4py import MPI
import time
import datetime
import numpy as np
from copy import deepcopy
import inspect

# Create a generic type variable
T = TypeVar('T')

# ================================================================
#  Print functions
# ================================================================

def print_bar(char='='):
    """
    Print a bar to the log file.

    Parameters
    char: str
        Character to use for the bar.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(char*80, flush=True)

def print_job_init_info():
    """
    Print the job starting time and the number of MPI processes.
    """
    print_bar("#")
    logger.info("FRIDOM: Framework for Idealized Ocean Models")
    # get system time
    from datetime import datetime

    # Get the current system time
    current_time = datetime.now()

    # Format the time according to the given format
    formatted_time = current_time.strftime(" > Job starting on %Y.%m.%d at %I:%M:%S %p")

    logger.info(formatted_time)

    # get the number of MPI processes
    size = MPI.COMM_WORLD.Get_size()
    logger.info(f" > Running on {size} MPI processes.")
    logger.info(f" > Backend: {config.backend}")
    print_bar("#")
    [print_bar(" ") for _ in range(3)]

# ================================================================
#  Formatting functions
# ================================================================

def humanize_number(value, unit):
    if unit == "meters":
        if value < 1e-2:
            return f"{value*1e3:.2f} mm"
        elif value < 1:
            return f"{value*1e2:.2f} cm"
        elif value < 1e3:
            return f"{value:.2f} m"
        else:
            return f"{value/1e3:.2f} km"

    elif unit == "seconds":
        delta = datetime.timedelta(seconds=float(value))
        days = delta.days
        formatted_time = ""
        if days > 0:
            years, days = divmod(days, 365)
            if years > 0:
                formatted_time += f"{years}y "
            if days > 0:
                formatted_time += f"{days}d "

        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = delta.microseconds // 1000
        microseconds = delta.microseconds % 1000

        if hours > 0 or days > 0:
            formatted_time += f"{hours:02d}:"
        if minutes > 0 or hours > 0 or days > 0:
            formatted_time += f"{minutes:02d}:"
        if seconds > 0 or minutes > 0 or hours > 0 or days > 0:
            formatted_time += f"{seconds:02d}s "
        if milliseconds > 0 or microseconds > 0:
            formatted_time += f"{milliseconds}"
            if microseconds > 0:
                formatted_time += f".{microseconds}"
            formatted_time += "ms"
        return formatted_time.strip()

    else:
        raise NotImplementedError(f"Unit '{unit}' not implemented.")

# ================================================================
#  Directory functions
# ================================================================

def chdir_to_submit_dir():
    """
    Change the current working directory to the directory where the job was submitted.
    """
    import os
    logger.info("Changing working directory")
    logger.info(f"Old working directory: {os.getcwd()}")
    submit_dir = os.getenv('SLURM_SUBMIT_DIR')
    os.chdir(submit_dir)
    logger.info(f"New working directory: {os.getcwd()}")
    return

def stdout_is_file():
    import os, sys
    # check if the output is not a file
    if os.isatty(sys.stdout.fileno()):
        res = False  # output is a terminal
    else:
        res = True   # output is a file

    # check if the output is ipython
    from IPython import get_ipython
    if get_ipython() is not None:
        res = False  # output is ipython
    return res


# ================================================================
#  Array functions
# ================================================================
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
    if config.backend_is_jax:
        return arr.at[where].set(value)
    else:
        res = arr.copy()
        res[where] = value
        return res
    
def random_array(shape: tuple[int], seed=12345):
    if config.backend_is_jax:
        import jax
        key = jax.random.key(seed)
        return jax.random.normal(key, shape)
    else:
        ncp = config.ncp
        default_rng = ncp.random.default_rng
        return default_rng(seed).standard_normal(shape)


class SliceableAttribute:
    """
    Class to make an object sliceable.
    
    Parameters
    ----------
    `slicer` : `callable`
        The slicer function.
    """
    def __init__(self, slicer: callable):
        self.slicer = slicer

    def __getitem__(self, key):
        return self.slicer(key)
# ================================================================
#  Numpy Conversion functions
# ================================================================
def _create_numpy_copy(obj, memo):
    # if the object has a __to_numpy__ method, call it
    if hasattr(obj, '__to_numpy__'):
        return obj.__to_numpy__(memo)

    # if the object has a _cpu attribute which is not None, return it
    if hasattr(obj, '_cpu'):
        if obj._cpu is not None:
            return obj._cpu

    # if the object is a cupy array, convert it to numpy and return it
    if isinstance(obj, config.ncp.ndarray):
        match config.backend:
            case config.Backend.NUMPY:
                return deepcopy(obj)
            case config.Backend.CUPY:
                return config.ncp.asnumpy(obj)
            case config.Backend.JAX_CPU:
                return np.array(obj)
            case config.Backend.JAX_GPU:
                return np.array(obj)

    # if the object is a numpy generic, return it
    if isinstance(obj, (np.ndarray, np.generic)):
        return deepcopy(obj)

    # if the object is a module, return it
    if inspect.ismodule(obj):
        return obj

    # if the object is a function, return it
    if inspect.isfunction(obj):
        return obj

    # if the object is a method, return it
    if inspect.ismethod(obj):
        return obj

    # if the object is a dictionary, convert all values
    if isinstance(obj, dict):
        return {key: to_numpy(value, memo) for key, value in obj.items()}

    # if the object is a list, convert all elements
    if isinstance(obj, list):
        return [to_numpy(x, memo) for x in obj]

    # if the object is a tuple, convert all elements
    if isinstance(obj, tuple):
        return tuple(to_numpy(x, memo) for x in obj)

    # if the object is a set, convert all elements
    if isinstance(obj, set):
        return {to_numpy(x, memo) for x in obj}

    # if the object is a type, return it
    if isinstance(obj, type):
        return deepcopy(obj)

    # if the object is a MPI.Cartcomm, return it
    if isinstance(obj, MPI.Cartcomm):
        return obj

    # if the object is not a python object, return a deepcopy
    if not hasattr(obj, '__dict__'):
        return deepcopy(obj)
    
    # if the object is a python object, convert all attributes
    d = id(obj)
    memo[d] = deepcopy(obj)
    for key, value in vars(obj).items():
        setattr(memo[d], key, to_numpy(value, memo))
    return memo[d]

def to_numpy(obj, memo=None, _nil=[]):
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
    # if the backend is numpy, return a deepcopy
    if config.backend == 'numpy':
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
        obj._cpu = memo[d]

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
    if isinstance(t, (np.datetime64, np.timedelta64)):
        return float(t.astype('timedelta64[s]').astype(float))
    return t

# ================================================================
#  JAX functions
# ================================================================

def jaxjit(fun: callable, *args, **kwargs) -> callable:
    """
    Decorator for JAX JIT compilation.
    
    Description
    -----------
    This decorator is a wrapper around jax.jit. When jax is not installed,
    the function is returned as it is.
    
    Parameters
    ----------
    `fun` : `callable`
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
    config.jax_jit_was_called = True
    if not config.enable_jax_jit:
        return fun

    if config.backend_is_jax:
        try:
            import jax
            return jax.jit(fun, *args, **kwargs)
        except ImportError:
            return fun
    else:
        return fun

def free_memory():
    """
    This function deletes all live buffers in the JAX backend.

    Description
    -----------
    This function destroys all live buffers in the JAX backend. This is
    useful for rerunning the code in the same session without running out
    of memory. 
    Note that the memory is only freed within JAX, not in the operating
    system. The operating system will still show the same memory usage.
    """
    if config.backend_is_jax:
        import jax
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers(): buf.delete()
    return

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
        The `dynamic` argument must be a tuple of attribute names. If you only
        have one dynamic attribute, use dynamic=('attr',) instead of dynamic=('attr').

    .. note::
        If a static attribute is changed, all jit compiled functions of the class
        must be recompiled. Hence, such attributes should be marked as dynamic.
        However, marking an attribute as dynamic will increase the computational
        cost. So, it is advisable to only mark attributes as dynamic that are
        actually changing during the simulation.

    .. warning::
        Methods that are jit compiled with fr.utils.jaxjit will not modify the
        object in place.
    
    Parameters
    ----------
    `cls` : `type`
        The class to add jax support to.
    `dynamic` : `tuple[str] | None` (default=None)
        A tuple of attribute names that should be considered dynamic.
    
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
    # if the backend is not jax, return the class as it is
    if not config.backend_is_jax:
        return cls
    import jax

    # make sure dynamic is either a tuple or None:
    if not isinstance(dynamic, (tuple, type(None))):
        config.logger.error(f"dynamic must be a tuple or None, not {type(dynamic)}")
        config.logger.error(f"In case you only have one dynamic attribute, ")
        config.logger.error(f"use dynamic=('attr',) instead of dynamic=('attr').")
        raise TypeError

    if dynamic is None:
        dynamic = []

    dynamic = list(dynamic) or []
    
    # check if the class has a _dynamic_attributes attribute
    if hasattr(cls, "_dynamic_jax_attrs"):
        dynamic += list(cls._dynamic_jax_attrs)

    # remove duplicates
    dynamic = set(dynamic)

    # set the new attributes
    cls._dynamic_jax_attrs = dynamic

    # define a function to flatten the class
    def _tree_flatten(self):
        # Store all attributes that are marked as dynamic
        children = tuple(getattr(self, attr) for attr in self._dynamic_jax_attrs)
    
        # Store all other attributes as aux_data
        aux_data = {key: att for key, att in self.__dict__.items() 
                    if key not in self._dynamic_jax_attrs}
    
        return (children, aux_data)

    # define a function to unflatten the class
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        # set dynamic attributes
        for i, attr in enumerate(cls._dynamic_jax_attrs):
            setattr(obj, attr, children[i])
        # set static attributes
        for key, value in aux_data.items():
            setattr(obj, key, value)
        return obj

    # set the new method to the class
    cls._tree_unflatten = _tree_unflatten

    # register the class with jax
    jax.tree_util.register_pytree_node(cls, _tree_flatten, cls._tree_unflatten)

    return cls

