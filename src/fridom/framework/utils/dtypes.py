"""Default data types derived from the JAX configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def dtype_real() -> type[np.floating]:
    """
    Return the default data type for real arrays.

    Description
    -----------
    The default real data type follows the JAX x64 configuration:
    float64 when x64 is enabled (the fridom default), float32 otherwise.
    To switch to single precision, disable x64 after importing fridom:

    .. code-block:: python

        import jax
        jax.config.update("jax_enable_x64", False)

    Returns
    -------
    type[np.floating]
        The default real scalar type.
    """
    return jnp.result_type(float).type


def dtype_comp() -> type[np.complexfloating]:
    """
    Return the default data type for complex arrays.

    Description
    -----------
    The default complex data type follows the JAX x64 configuration:
    complex128 when x64 is enabled (the fridom default), complex64
    otherwise.

    Returns
    -------
    type[np.complexfloating]
        The default complex scalar type.
    """
    return jnp.result_type(complex).type
