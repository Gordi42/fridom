"""
Configurations for the framework.

Description
-----------
This module contains the basic configurations for the framework, for example the
default backend (numpy or cupy), the default data types (real and complex), etc.

Attributes
----------
`backend` : `str`
    The current backend used for computations (numpy or cupy).
`ncp` : `module`
    The current numpy-like module used for computations (numpy or cupy).
`scp` : `module`
    The current scipy-like module used for computations (scipy or cupyx.scipy).
`dtype_real` : `np.dtype`
    The current default data type for real numbers.
`dtype_comp` : `np.dtype`
    The current default data type for complex numbers.

Functions
---------
`set_backend(new_backend: str)`
    Set the backend to use for computations (numpy or cupy).
`set_dtype_real(new_dtype_real: np.dtype)`
    Set the default real data type.
`set_dtype_comp(new_dtype_comp: np.dtype)`
    Set the default complex data type.

Examples
--------
>>> import fridom.framework as fr
>>> fr.config.set_backend("numpy")
>>> print(fr.config.ncp)
<module 'numpy' from '.../numpy/__init__.py'>
>>> fr.config.set_backend("cupy")
>>> print(fr.config.ncp)
<module 'cupy' from '.../cupy/__init__.py'>
"""
import numpy as np

# -----------------------------------------------------------------------------
#  Global configurations
# -----------------------------------------------------------------------------
backend = None
ncp = None  # numpy or cupy
scp = None  # scipy or cupyx.scipy
dtype_real = np.float64
dtype_comp = np.complex128

# -----------------------------------------------------------------------------
#  Setters
# -----------------------------------------------------------------------------

def _set_default_backend():
    """
    Set the default backend (cupy if available, numpy otherwise).
    """
    # get ncp and backend from global scope
    global ncp
    global scp
    global backend
    try:  # Try to import cupy
        import cupy
        import cupyx.scipy
        backend = "cupy"
        ncp = cupy
        scp = cupyx.scipy
    except ImportError:  # If cupy is not available, use numpy instead
        import numpy
        import scipy
        backend = "numpy"
        ncp = numpy
        scp = scipy
    return

def set_backend(new_backend: str):
    """
    Set the backend to use for computations (numpy or cupy).
    
    Parameters
    ----------
    `new_backend` : `str`
        The new backend to use for computations (numpy or cupy)
    
    Raises
    ------
    `ValueError`
        Unsupported backend.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_backend("numpy")
    >>> print(fr.config.ncp)
    <module 'numpy' from '.../numpy/__init__.py'>
    >>> fr.config.set_backend("cupy")
    >>> print(fr.config.ncp)
    <module 'cupy' from '.../cupy/__init__.py'>
    """
    global ncp
    global scp
    global backend
    if new_backend == "numpy":
        import numpy
        import scipy
        ncp = numpy
        scp = scipy
    elif new_backend == "cupy":
        import cupy
        import cupyx.scipy
        ncp = cupy
        scp = cupyx.scipy
    else:
        raise ValueError(f"Backend {new_backend} not supported.")
    backend = new_backend
    return

def set_dtype_real(new_dtype_real: np.dtype):
    """
    Set the default real data type.
    
    Parameters
    ----------
    `new_dtype_real` : `np.dtype`
        The new default real data type.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_dtype_real(np.float32)
    >>> print(fr.config.dtype_real)
    <class 'numpy.float32'>
    """
    global dtype_real
    dtype_real = new_dtype_real
    return

def set_dtype_comp(new_dtype_comp: np.dtype):
    """
    Set the default complex data type.
    
    Parameters
    ----------
    `new_dtype_comp` : `np.dtype`
        The new default complex data type.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_dtype_comp(np.complex64)
    >>> print(fr.config.dtype_comp)
    <class 'numpy.complex64'>
    """
    global dtype_comp
    dtype_comp = new_dtype_comp
    return


# =============================================================================
#  Default configs
# =============================================================================
_set_default_backend()