import pytest
import fridom.framework as fr
import numpy as np

def test_set_backend(backend):
    assert fr.config.backend == backend
    match backend:
        case "numpy":
            assert fr.config.ncp.__name__ == "numpy"
        case "cupy":
            assert fr.config.ncp.__name__ == "cupy"
        case "jax_cpu":
            assert fr.config.ncp.__name__ == "jax.numpy"
        case "jax_gpu":
            assert fr.config.ncp.__name__ == "jax.numpy"
    # if the backend is JAX, check if the device is correct
    if fr.config.backend_is_jax:
        from jax import extend
        device = extend.backend.get_backend().platform
        match backend:
            case "jax_cpu":
                assert device == "cpu"
            case "jax_gpu":
                assert device == "gpu"


@pytest.mark.parametrize(
    "dtype, expected_real, expected_comp",
    [
        ("float32", np.float32, np.complex64),
        ("float64", np.float64, np.complex128),
        ("float128", np.float128, np.complex256),
    ],
)
def test_set_dtype(backend, dtype, expected_real, expected_comp):
    # JAX does not support float128 so we skip this test
    if fr.config.backend_is_jax and dtype == "float128":
        return
    # We first save the original dtype to reset the config after the test
    dtype_original = fr.config.dtype_real
    fr.config.set_dtype(dtype)
    assert fr.config.dtype_real == expected_real
    assert fr.config.dtype_comp == expected_comp
    # Reset the config to the original dtype
    fr.config.set_dtype(dtype_original)
