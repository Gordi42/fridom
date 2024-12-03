"""tests for framework/config.py"""
import pytest
import numpy as np
import fridom.framework as fr


@pytest.fixture(autouse=True)
def reset_config():
    """Fixture to reset the config after each test."""
    original_dtype = fr.config.dtype_real
    yield
    fr.config.set_dtype(original_dtype)

@pytest.mark.parametrize(
    "dtype, expected_real, expected_comp",
    [
        ("float32", np.float32, np.complex64),
        ("float64", np.float64, np.complex128),
        ("float128", np.float128, np.complex256),
    ],
)
def test_set_dtype(dtype, expected_real, expected_comp):
    """Test setting data types and ensuring compatibility."""
    if fr.config.backend_is_jax and dtype == "float128":
        pytest.skip("JAX does not support float128.")

    fr.config.set_dtype(dtype)
    assert fr.config.dtype_real == expected_real
    assert fr.config.dtype_comp == expected_comp


def test_enable_jax_jit_flag():
    """Test toggling the enable_jax_jit flag."""
    original_state = fr.config.enable_jax_jit

    fr.config.enable_jax_jit = not original_state
    assert fr.config.enable_jax_jit != original_state

    # Reset the state
    fr.config.enable_jax_jit = original_state
    assert fr.config.enable_jax_jit == original_state
