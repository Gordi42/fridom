import pytest
import fridom.framework as fr
import numpy as np

def test_set_backend(backend):
    assert fr.config.backend == backend
    match backend:
        case fr.config.Backend.NUMPY:
            assert fr.config.ncp.__name__ == "numpy"
        case fr.config.Backend.CUPY:
            assert fr.config.ncp.__name__ == "cupy"
        case fr.config.Backend.JAX_CPU:
            assert fr.config.ncp.__name__ == "jax.numpy"
        case fr.config.Backend.JAX_GPU:
            assert fr.config.ncp.__name__ == "jax.numpy"
    # if the backend is JAX, check if the device is correct
    if fr.config.backend_is_jax:
        import jax
        device = jax.lib.xla_bridge.get_backend().platform
        match backend:
            case fr.config.Backend.JAX_CPU:
                assert device == "cpu"
            case fr.config.Backend.JAX_GPU:
                assert device == "gpu"


@pytest.mark.parametrize("dtype", [fr.config.DType.FLOAT32, 
                                   fr.config.DType.FLOAT64])
def test_set_dtype(dtype):
    fr.config.set_dtype(dtype)
    match dtype:
        case fr.config.DType.FLOAT32:
            assert fr.config.dtype_real == np.float32
            assert fr.config.dtype_comp == np.complex64
        case fr.config.DType.FLOAT64:
            assert fr.config.dtype_real == np.float64
            assert fr.config.dtype_comp == np.complex128
        case fr.config.DType.FLOAT128:
            assert fr.config.dtype_real == np.float128
            assert fr.config.dtype_comp == np.complex256
    return
