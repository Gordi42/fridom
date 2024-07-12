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
from enum import Enum
import logging
import sys
import os
from IPython import get_ipython
import coloredlogs

# ================================================================
#  BACKEND
# ================================================================

backend = None
ncp = None  # numpy or cupy
scp = None  # scipy or cupyx.scipy

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

# ================================================================
#  DATA TYPES
# ================================================================

dtype_real = np.float64
dtype_comp = np.complex128

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
            '%(levelname)s: %(message)s')
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
    return


# =============================================================================
#  Default configs
# =============================================================================
_set_default_backend()
_setup_logging()