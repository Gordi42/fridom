"""
utils
===
Contains utility functions.

Functions
---------
`print_bar(char='=')`
    Print a bar to the stdout.
`print_job_init_info()`
    Print the job starting time and the number of MPI processes.
"""
from . import config
from .config import logger
from mpi4py import MPI
import time
import numpy as np

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
#  Progress bar class
# ================================================================
class ProgressBar:
    """
    Progress bar class.
    
    Description
    -----------
    The progress bar class is a wrapper around the tqdm progress bar. It
    has a custom format and handles the output to the stdout when the
    stdout is a file.
    
    Parameters
    ----------
    `disable` : `bool`
        Whether to disable the progress bar.
    
    Methods
    -------
    `update(value: float, postfix: str)`
        Updates the progress bar.
    `close()`
        Close the progress bar.
    """
    def __init__(self, disable: bool = False) -> None:
        # only rank 0 should print the progress bar
        if MPI.COMM_WORLD.Get_rank() != 0:
            disable = True
        # ----------------------------------------------------------------
        #  Set the progress bar format
        # ----------------------------------------------------------------
        bar_format = "{percentage:3.2f}%|{bar}| "
        bar_format += "[{elapsed}<{remaining}]{postfix}"

        # ----------------------------------------------------------------
        #  Check if the stdout is a file
        # ----------------------------------------------------------------
        file_output = stdout_is_file()
        if file_output:
            # if the stdout is a file, tqdm would print to the stderr by default
            # we could instead print to the stdout, but this would mess up
            # the look of the progress bar due to "\r" characters
            # so we create a StringIO object to capture the output
            # and adjust the progress bar accordingly
            import io
            output = io.StringIO()
        else:
            import sys
            output = sys.stdout

        # ----------------------------------------------------------------
        #  Create the progress bar
        # ----------------------------------------------------------------
        from tqdm import tqdm
        pbar = tqdm(
            total=100, 
            disable=disable, 
            bar_format=bar_format, 
            unit="%", 
            file=output)
        
        # ----------------------------------------------------------------
        #  Set the attributes
        # ----------------------------------------------------------------
        self.disable = disable
        # private attributes
        self._pbar = pbar
        self._file_output = file_output
        self._output = output
        self._last_call = time.time()
        return

    def update(self, value: float, postfix: str = "") -> None:
        """
        Updates the progress bar.
        
        Parameters
        ----------
        `value` : `float`
            A value between 0 and 100, representing the progress.
        `postfix` : `str`
            A string to append to the progress bar.
        """
        if self.disable:
            return

        # get the time between the last call (in milliseconds)
        now = time.time()
        elapsed = now - self._last_call
        self._last_call = now
        elapsed = f"{int(elapsed*1e3)} ms/it"

        # update the progress bar
        self._pbar.n = value
        self._pbar.set_postfix_str(f"{elapsed}  at {postfix}")
        self._pbar.refresh()

        if not self._file_output:
            return

        # print the progress to the stdout
        config.logger.info(self._output.getvalue().split("\r")[1])

        # clear the output string
        self._output.seek(0)
        return

    def close(self) -> None:
        """Close the progress bar."""
        self._pbar.close()
        return

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
        arr[where] = value
        return arr
    
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


def _tree_flatten(self):
    # Store all attributes that are marked as dynamic
    children = tuple(getattr(self, attr) for attr in self._dynamic_attributes)
    
    # Store all other attributes as aux_data
    aux_data = {key: att for key, att in self.__dict__.items() 
                if key not in self._dynamic_attributes}
    
    return (children, aux_data)


@classmethod
def _tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    # set dynamic attributes
    for i, attr in enumerate(cls._dynamic_attributes):
        setattr(obj, attr, children[i])
    # set static attributes
    for key, value in aux_data.items():
        setattr(obj, key, value)
    return obj


def jaxify_class(cls: type) -> None:
    """
    Add JAX pytree support to a class (for jit compilation).
    
    Description
    -----------
    In order to use jax.jit on custom classes, the class must be registered
    to jax. This function adds the necessary methods to the class to make it
    compatible with jax.jit. 
    By default, all attributes of an object are considered static, i.e., they
    they will not be traced by jax. Attributes that should be dynamic must
    be marked in the class definition inside the `_dynamic_attributes` list
    (see example).
    
    Parameters
    ----------
    `cls` : `type`
        The class to add jax support to.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> class MyClass:
    ...     _dynamic_attributes = ["x",]
    ...     def __init__(self, x):
    ...         self.x = x
    ...         self.my_static_attribute = 42
    ...
    ...     @fr.utils.jaxjit
    ...     def my_method(self):
    ...         return self.x**2
    ...
    >>> fr.utils.jaxify_class(MyClass)
    """
    if config.backend_is_jax:
        try:
            import jax
            cls._tree_unflatten = _tree_unflatten
            jax.tree_util.register_pytree_node(cls, _tree_flatten, cls._tree_unflatten)
        except ImportError:
            pass
    return
