import pytest
import fridom.framework as fr
import numpy as np

def test_set_backend(backend):
    assert fr.config.backend == backend
    assert fr.config.ncp.__name__ == backend

@pytest.mark.parametrize("dtype_real", [np.float32, np.float64])
def test_set_dtype_real(dtype_real):
    fr.config.set_dtype_real(dtype_real)
    assert fr.config.dtype_real == dtype_real

@pytest.mark.parametrize("dtype_comp", [np.complex64, np.complex128])
def test_set_dtype_comp(dtype_comp):
    fr.config.set_dtype_comp(dtype_comp)
    assert fr.config.dtype_comp == dtype_comp