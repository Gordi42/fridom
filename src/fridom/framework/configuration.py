"""config.py - The configuration file for the fridom framework."""
from __future__ import annotations

import os
import time

import numpy  # noqa: ICN001
import scipy

from fridom.framework.logger import log


# We want to import the modules in the respective functions to avoid unnecessary
# imports. This will speed up the import time of the fridom framework.
class Config:

    """Configuration class for the fridom framework."""

    _ncp = numpy
    _scp = scipy
    _backend = "numpy"
    _enable_parallel = False
    _enable_jax_jit = True
    _dtype_real = numpy.float64
    _dtype_comp = numpy.complex128
    _load_time: float = 0

    def __init__(self) -> None:
        self._load_time = time.time()
        # Set the backend
        # check if the backend is set via an environment variable
        backend = os.getenv("FRIDOM_BACKEND", None)
        if backend is not None:
            try:
                self._set_backend_unsafe(backend)
            except (ImportError, RuntimeError):
                log.warning(
                    "Backend %s is not available. Falling back to default backend.",
                    backend,
                )
                return
            else:
                return
        # If no backend is set, we try to set the backend in the following order
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
    def __repr__(self) -> str:
        res = "Config(\n"
        res += f" - backend = {self.backend},\n"
        if self.backend_is_jax:
            res += f" - enable_jax_jit = {self.enable_jax_jit},\n"
        res += f" - dtype = {self.dtype_real},\n"
        res += ")"
        return res

    # ----------------------------------------------------------------
    #  Unsafe backend setters
    # ----------------------------------------------------------------
    @classmethod
    def _set_backend_unsafe(cls, backend_name: str) -> None:
        """Set the backend without checking if the backend is available."""
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
                message = f"Backend {backend_name} not supported."
                raise ValueError(message)

    @classmethod
    def _set_numpy_as_backend(cls) -> None:
        cls._ncp = numpy
        cls._scp = scipy
        cls._backend = "numpy"

    @classmethod
    def _set_cupy_as_backend_unsafe(cls) -> None:
        import cupy
        import cupyx.scipy

        cls._ncp = cupy
        cls._scp = cupyx.scipy
        cls._backend = "cupy"

    @classmethod
    def _set_jax_cpu_as_backend_unsafe(cls) -> None:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp

        cls._ncp = jnp
        cls._scp = jsp
        cls._backend = "jax_cpu"
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", val=True)

    @classmethod
    def _set_jax_gpu_as_backend_unsafe(cls) -> None:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        from jax import extend

        jax.config.update("jax_platform_name", "gpu")
        jax.config.update("jax_enable_x64", val=True)
        # the next line will raise a RuntimeError if the GPU is not available
        _ = extend.backend.get_backend().platform
        cls._ncp = jnp
        cls._scp = jsp
        cls._backend = "jax_gpu"

    # ----------------------------------------------------------------
    #  Data Types
    # ----------------------------------------------------------------
    @classmethod
    def set_dtype(cls, dtype: str | numpy.dtype) -> None:
        """
        Set the default data type for real and complex arrays.

        Parameters
        ----------
        dtype : str or numpy.dtype
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
        backend_is_jax = cls._backend.startswith("jax")
        if backend_is_jax and dtype == numpy.float128:
            log.warning(
                "float128 is not supported for the JAX backend. "
                "Falling back to float64.")
            dtype = numpy.dtype(numpy.float64)
        # set the new data types
        cls._dtype_real = dtype.type
        cls._dtype_comp = numpy.dtype(f"complex{dtype.itemsize * 16}").type

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def ncp(self):  # noqa: ANN201
        """Numpy-like backend."""
        return self._ncp

    @property
    def scp(self):  # noqa: ANN201
        """Scipy-like backend."""
        return self._scp

    @property
    def enable_parallel(self) -> bool:
        """Whether to enable parallelism."""
        return self._enable_parallel

    @enable_parallel.setter
    def enable_parallel(self, value: bool) -> bool:
        _ = value
        message = "Parallelism is not yet supported."
        raise ValueError(message)

    @property
    def backend(self) -> str:
        """The current backend."""
        return self._backend

    @property
    def backend_is_jax(self) -> bool:
        """Check if the backend is JAX."""
        return self._backend.startswith("jax")

    @property
    def enable_jax_jit(self) -> bool:
        """Whether to enable jax.jit."""
        return self._enable_jax_jit

    @enable_jax_jit.setter
    def enable_jax_jit(self, value: bool) -> None:
        self._enable_jax_jit = value

    @property
    def dtype_real(self) -> numpy.dtype:
        """The default dtype of real arrays."""
        return self._dtype_real

    @property
    def dtype_comp(self) -> numpy.dtype:
        """The default dtype of complex arrays."""
        return self._dtype_comp

    @property
    def load_time(self) -> float:
        """The time when the config was loaded."""
        return self._load_time


config = Config()
