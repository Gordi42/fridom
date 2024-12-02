"""config.py - The configuration file for the fridom framework"""
import numpy
import scipy
from fridom.framework.logger import log


# We want to import the modules in the respective functions to avoid unnecessary
# imports. This will speed up the import time of the fridom framework.
# pylint: disable=import-outside-toplevel
# pylint: disable=import-error
class Config:
    """Configuration class for the fridom framework."""
    _ncp = numpy
    _scp = scipy
    _backend = "numpy"
    _jax_jit_was_called = False
    _dtype_real = numpy.float64
    _dtype_comp = numpy.complex128

    def __init__(self):
        backend_try_order = ["jax_gpu", "jax_cpu", "cupy", "numpy"]
        # we try to set the backend in the order of the backend_try_order list
        for backend in backend_try_order:
            try:
                self._set_backend_unsafe(backend)
                break
            except ImportError:
                pass
            except RuntimeError:
                pass

    # ----------------------------------------------------------------
    #  Representation
    # ----------------------------------------------------------------
    def __repr__(self):
        res = "Config(\n"
        res += f" - backend = {self.backend},\n"
        res += f" - jax_jit_was_called = {self.jax_jit_was_called}\n"
        res += f" - dtype = {self.dtype_real},\n"
        res += ")"
        return res

    # ----------------------------------------------------------------
    #  Unsafe backend setters
    # ----------------------------------------------------------------
    @classmethod
    def _set_backend_unsafe(cls, backend_name: str):
        """Sets the backend without checking if the backend is available."""
        match backend_name:
            case "numpy":
                cls._set_numpy_as_backend()
            case "cupy":
                cls._set_cupy_as_backend_unsafe()
            case "jax_cpu":
                cls._set_jax_cpu_as_backend_unsafe()
            case "jax_gpu":
                cls._set_jax_gpu_as_backend_unsafe()
            case _:
                raise ValueError(f"Backend {backend_name} not supported.")

    @classmethod
    def _set_numpy_as_backend(cls):
        cls._ncp = numpy
        cls._scp = scipy
        cls._backend = "numpy"

    @classmethod
    def _set_cupy_as_backend_unsafe(cls):
        import cupy
        import cupyx.scipy
        cls._ncp = cupy
        cls._scp = cupyx.scipy
        cls._backend = "cupy"

    @classmethod
    def _set_jax_cpu_as_backend_unsafe(cls):
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        cls._ncp = jnp
        cls._scp = jsp
        cls._backend = "jax_cpu"
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_enable_x64', True)

    @classmethod
    def _set_jax_gpu_as_backend_unsafe(cls):
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        from jax import extend
        jax.config.update('jax_platform_name', 'gpu')
        jax.config.update('jax_enable_x64', True)
        # the next line will raise a RuntimeError if the GPU is not available
        _ = extend.backend.get_backend().platform
        cls._ncp = jnp
        cls._scp = jsp
        cls._backend = "jax_gpu"

    # ----------------------------------------------------------------
    #  Safe backend setters
    # ----------------------------------------------------------------
    @classmethod
    def set_backend(cls, backend_name: str):
        """
        Set the backend to use for computations (numpy like)
    
        Parameters
        ----------
        `new_backend` : str
            The new backend to use for computations. The following backends are
            supported:
            - "numpy"
            - "cupy"
            - "jax_cpu"
            - "jax_gpu"
        `silent` : bool, optional (default=False)
            If True, no warning will be printed if the backend is changed after
            calling `jax.jit`.
    
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

        # print a warning if the backend is changed after jax.jit was called
        if backend_name != cls._backend and cls.jax_jit_was_called:
            log.warning(
                "jax.jit was called before setting the backend. "
                "This might lead to unexpected behavior.")

        match backend_name:
            case "numpy":
                cls._set_numpy_as_backend()
            case "cupy":
                cls._set_cupy_as_backend()
            case "jax_cpu":
                cls._set_jax_cpu_as_backend()
            case "jax_gpu":
                cls._set_jax_gpu_as_backend()
            case _:
                raise ValueError(f"Backend {backend_name} not supported.")

    @classmethod
    def _set_cupy_as_backend(cls):
        try:
            cls._set_cupy_as_backend_unsafe()
        except ImportError:
            log.error("Failed to import cupy. Falling back to numpy.")
            cls.set_backend("numpy")

    @classmethod
    def _set_jax_cpu_as_backend(cls):
        try:
            cls._set_jax_cpu_as_backend_unsafe()
        except ImportError:
            log.error("Failed to import jax. Falling back to numpy.")
            cls.set_backend("numpy")

    @classmethod
    def _set_jax_gpu_as_backend(cls):
        try:
            cls._set_jax_gpu_as_backend_unsafe()
        except ImportError:
            log.error("Failed to import jax. Falling back to numpy.")
            cls.set_backend("numpy")
        except RuntimeError:
            log.error("GPU not available. Falling back to JAX_CPU.")
            cls.set_backend("jax_cpu")

    # ----------------------------------------------------------------
    #  Data Types
    # ----------------------------------------------------------------
    @classmethod
    def set_dtype(cls, dtype: str | numpy.dtype):
        """
        Set the default data type for real and complex arrays.
    
        Parameters
        ----------
        `dtype` : str or numpy.dtype
            The new default data type for real arrays. Complex arrays will be
            set so that both real and imaginary parts have the same data type.
            Available data types are:
            - "float32"
            - "float64"
            - "float128"

        Examples
        --------
        >>> import fridom.framework as fr
        >>> fr.config.set_dtype("float32")
        >>> print(fr.config.dtype_real)
        dtype('float32')
        >>> print(fr.config.dtype_comp)
        dtype('complex64')
        """
        dtype = numpy.dtype(dtype)
        # for the gpu backend, float128 is not supported
        if cls.backend_is_jax and dtype == numpy.float128:
            log.warning("float128 is not supported for the JAX backend. "
                        "Falling back to float64.")
            dtype = numpy.dtype(numpy.float64)
        # set the new data types
        cls._dtype_real = dtype
        cls._dtype_comp = numpy.dtype(f"complex{dtype.itemsize * 16}")

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def ncp(self):
        """Numpy-like backend."""
        return self._ncp

    @property
    def scp(self):
        """Scipy-like backend."""
        return self._scp

    @property
    def backend(self) -> str:
        """The current backend."""
        return self._backend

    @property
    def backend_is_jax(self) -> bool:
        """Check if the backend is JAX."""
        return self._backend.startswith("jax")

    @property
    def jax_jit_was_called(self) -> bool:
        """Check if jax.jit was called."""
        return self._jax_jit_was_called

    @jax_jit_was_called.setter
    def jax_jit_was_called(self, value: bool):
        """Set the jax_jit_was_called flag."""
        self._jax_jit_was_called = value

    @property
    def dtype_real(self) -> numpy.dtype:
        """The default dtype of real arrays."""
        return self._dtype_real

    @property
    def dtype_comp(self) -> numpy.dtype:
        """The default dtype of complex arrays."""
        return self._dtype_comp


config = Config()
