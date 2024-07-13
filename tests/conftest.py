import pytest
import fridom.framework as fr

# Fixture to enable GPU testing

try:
    import cupy
    cupy_unavailable = False
except ImportError:
    cupy_unavailable = True

@pytest.fixture(scope='module', params=[
    pytest.param(fr.config.Backend.NUMPY, id="numpy"),
    pytest.param(fr.config.Backend.CUPY, id="cupy", marks=pytest.mark.skipif(
        cupy_unavailable, reason="Cupy not available"))])
def backend(request):
    fr.config.set_backend(request.param)
    return request.param