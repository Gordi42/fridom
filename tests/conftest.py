import pytest
import fridom.framework as fr

# Fixture to enable GPU testing

try:
    import cupy
    cupy_unavailable = False
except ImportError:
    cupy_unavailable = True
# check if JAX is available
try:
    import jax
    jax_unavailable = False
except ImportError:
    jax_unavailable = True
jax_backend = fr.config.Backend.JAX_GPU

@pytest.fixture(scope='module', params=[
    pytest.param(fr.config.Backend.NUMPY),
    pytest.param(fr.config.Backend.CUPY, marks=pytest.mark.skipif(
        cupy_unavailable, reason="Cupy not available")),
    pytest.param(jax_backend, marks=pytest.mark.skipif(
        jax_unavailable, reason="Jax not available"))
])
def backend(request):
    fr.config.set_backend(request.param)
    return request.param