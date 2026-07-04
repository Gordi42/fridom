"""tests for framework/utils/dtypes.py."""
import jax
import numpy as np

import fridom.framework as fr


def test_default_dtypes():
    """By default (x64 enabled), fridom uses double precision."""
    assert fr.utils.dtype_real() == np.float64
    assert fr.utils.dtype_comp() == np.complex128


def test_single_precision_dtypes():
    """Disabling x64 switches the default dtypes to single precision."""
    try:
        jax.config.update("jax_enable_x64", val=False)
        assert fr.utils.dtype_real() == np.float32
        assert fr.utils.dtype_comp() == np.complex64
    finally:
        jax.config.update("jax_enable_x64", val=True)
