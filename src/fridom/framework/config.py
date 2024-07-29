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
from enum import Enum, auto
import logging
import sys
import os
from IPython import get_ipython
import coloredlogs
import time

# ================================================================
#  Store the time when the module is loaded
# ================================================================
load_time = time.time()

# ================================================================
#  LOGGING
# ================================================================
class LogLevel(Enum):
    """
    Logging Levels:
    ---------------
    10 - DEBUG:
        The lowest logging level. It is only used for debugging purposes.
        Computational expensive debugging information should be logged at this level.
        For debugging purposes: everything is logged.
    15 - VERBOSE:
        The lowest logging level that is not used for debugging. This should
        be used for detailed documentation of what is happening in the code.
    20 - INFO:
        This should be used for general information.
    25 - NOTICE:
        This should be used for the very essential information.
    30 - SILENT:
        No information is logged (but warnings and errors are still logged).
    40 - ERROR:
        No warnings are logged, only errors.
    50 - CRITICAL:
        Only critical errors are logged.
    """
    DEBUG = 10
    VERBOSE = 15
    INFO = 20
    NOTICE = 25
    SILENT = 30
    ERROR = 40
    CRITICAL = 50

logger = logging.getLogger("fridom")


def set_log_level(level: LogLevel | int):
    """
    Set the logging level.
    
    Parameters
    ----------
    `level` : `LogLevel`
        The logging level.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_log_level(fr.LogLevel.INFO)
    """
    global logger
    if isinstance(level, LogLevel):
        logger.setLevel(level.value)
    else:
        logger.setLevel(level)
    return

def set_log_ranks(ranks: list[int] | None):
    """
    Set the ranks which should log information.
    
    Parameters
    ----------
    `ranks` : `list[int] | None`
        A list of MPI ranks which should log the information. If None, all ranks log.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_log_ranks([0, 1])
    """
    global logger
    if ranks is None:
        logger.disabled = False
        return
    from mpi4py import MPI
    my_rank = MPI.COMM_WORLD.Get_rank()
    if my_rank in ranks:
        logger.disabled = False
    else:
        logger.disabled = True
    return

def _setup_logging():
    """
    Setup the logging configuration.
    """
    global logger
    # add logging level names
    logging.addLevelName(LogLevel.VERBOSE.value, "VERBOSE")
    logging.addLevelName(LogLevel.NOTICE.value, "NOTICE")

    # logger = logging.getLogger("Fridom Logger")
    console_handler = logging.StreamHandler(stream=sys.stdout)

    # check if the output should be colorized
    if os.isatty(sys.stdout.fileno()):
        _colored_output = True  # colors in terminal
    else:
        _colored_output = False  # no colors in file
    if get_ipython() is not None:
        _colored_output = True  # colors in ipython

    if _colored_output:
        formatter = coloredlogs.ColoredFormatter(
            '%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    else:
        formatter = logging.Formatter(
            '%(asctime)s: %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    set_log_level(LogLevel.SILENT)

    # add verbose and notice methods
    def verbose(message: str, *args, **kwargs) -> None:
        if logger.isEnabledFor(LogLevel.VERBOSE.value):
            logger._log(LogLevel.VERBOSE.value, message, args, **kwargs)

    def notice(message: str, *args, **kwargs) -> None:
        if logger.isEnabledFor(LogLevel.NOTICE.value):
            logger._log(LogLevel.NOTICE.value, message, args, **kwargs)
    
    logger.verbose = verbose
    logger.notice = notice

    set_log_ranks([0])
    return

_setup_logging()

# ================================================================
#  BACKEND
# ================================================================
class Backend(Enum):
    """
    Backend Types:
    ---------------
    `NUMPY`
        Use numpy as the backend.
    `CUPY`
        Use cupy as the backend.
    `JAX_CPU`
        Use jax with CPU as the backend.
    `JAX_GPU`
        Use jax with GPU as the backend.
    """
    NUMPY = auto()
    CUPY = auto()
    JAX_CPU = auto()
    JAX_GPU = auto()

backend: Backend = None
backend_is_jax: bool
enable_jax_jit: bool = True
jax_jit_was_called: bool = False
ncp = None  # numpy or cupy
scp = None  # scipy or cupyx.scipy
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def set_backend(new_backend: Backend, silent: bool = False):
    """
    Set the backend to use for computations (numpy like)
    
    Parameters
    ----------
    `new_backend` : `Backend`
        The new backend to use for computations
    
    Raises
    ------
    `ValueError`
        Unsupported backend.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_backend(fr.Backend.NUMPY)
    >>> print(fr.config.ncp)
    <module 'numpy' from '.../numpy/__init__.py'>
    >>> fr.config.set_backend(fr.Backend.CUPY)
    >>> print(fr.config.ncp)
    <module 'cupy' from '.../cupy/__init__.py'>
    """
    global ncp
    global scp
    global backend
    global backend_is_jax
    global logger
    global jax_jit_was_called

    # print a warning if the backend is changed after jax.jit was called
    if backend is not None:
        if new_backend != backend and jax_jit_was_called and not silent:
            logger.warning(
                "jax.jit was called before setting the backend. "
                "This might lead to unexpected behavior.")

    match new_backend:
        case Backend.NUMPY:
            import numpy
            import scipy
            backend_is_jax = False
            ncp = numpy
            scp = scipy
        case Backend.CUPY:
            try:
                import cupy
                import cupyx.scipy
                backend_is_jax = False
                ncp = cupy
                scp = cupyx.scipy
            except ImportError:
                logger.error("Failed to import cupy. Falling back to numpy.")
                set_backend(Backend.NUMPY)
                return
        case Backend.JAX_CPU:
            try:
                import jax
                import jax.numpy
                import jax.scipy
                backend_is_jax = True
                ncp = jax.numpy
                scp = jax.scipy
                jax.config.update('jax_platform_name', 'cpu')
                jax.config.update('jax_enable_x64', True)
            except ImportError:
                logger.error("Failed to import jax. Falling back to numpy.")
                set_backend(Backend.NUMPY)
                return
        case Backend.JAX_GPU:
            try:
                import jax
                import jax.numpy
                import jax.scipy
                backend_is_jax = True
                ncp = jax.numpy
                scp = jax.scipy
                jax.config.update('jax_platform_name', 'gpu')
                jax.config.update('jax_enable_x64', True)
                # the next line will raise a RuntimeError if the GPU is not available
                jax.lib.xla_bridge.get_backend().platform
            except ImportError:
                logger.error("Failed to import jax. Falling back to cupy.")
                set_backend(Backend.CUPY)
                return
            except RuntimeError:
                logger.error("GPU not available. Falling back to JAX_CPU.")
                set_backend(Backend.JAX_CPU)
                return
        case _:
            raise ValueError(f"Backend {new_backend} not supported.")
    backend = new_backend
    return

def _set_default_backend():
    """
    Set the default backend (cupy if available, numpy otherwise).
    """
    # disable the logger temporarily
    logger_disabled = logger.disabled
    logger.disabled = True
    set_backend(Backend.CUPY)
    # restore the logger state
    logger.disabled = logger_disabled
    return

_set_default_backend()

# ================================================================
#  DATA TYPES
# ================================================================
class DType(Enum):
    """
    Data Types:
    -----------
    `FLOAT32`
        32-bit floating point number.
    `FLOAT64`
        64-bit floating point number.
    `FLOAT128`
        128-bit floating point number.

    Note:
    -----
    Complex numbers are twice the size of real numbers.
    """
    FLOAT32 = auto()
    FLOAT64 = auto()
    FLOAT128 = auto()

dtype_real = np.float64
dtype_comp = np.complex128

def set_dtype(dtype: DType):
    """
    Set the default data type.
    
    Parameters
    ----------
    `dtype` : `DType`
        The new default data type of real values.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> fr.config.set_dtype(fr.DType.FLOAT32)
    >>> print(fr.config.dtype_real)
    <class 'numpy.float32'>
    """
    global dtype_real
    global dtype_comp
    global backend
    match dtype:
        case DType.FLOAT32:
            dtype_real = np.float32
            dtype_comp = np.complex64
        case DType.FLOAT64:
            dtype_real = np.float64
            dtype_comp = np.complex128
        case DType.FLOAT128:
            if backend == Backend.JAX_CPU or backend == Backend.JAX_GPU:
                logger.warning("JAX does not support FLOAT128, using FLOAT64 instead.")
                set_dtype(DType.FLOAT64)
                return
            dtype_real = np.float128
            dtype_comp = np.complex256
        case _:
            raise ValueError(f"Data type {dtype} not supported.")