import pytest
import fridom.framework as fr

# Fixture to enable GPU testing

try:
    import cupy
    cupy_unavailable = False
except ImportError:
    cupy_unavailable = True

@pytest.fixture(scope='module', params=[
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy", marks=pytest.mark.skipif(
        cupy_unavailable, reason="Cupy not available"))])
def backend(request):
    fr.config.set_backend(request.param)
    return request.param