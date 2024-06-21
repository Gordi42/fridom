import pytest

# Fixture to enable GPU testing

try:
    import cupy
    gpu_unavailable = False
except ImportError:
    gpu_unavailable = True

@pytest.fixture(scope='module', params=[
    pytest.param(False, id="CPU"),
    pytest.param(True, id="GPU", marks=pytest.mark.skipif(
        gpu_unavailable, reason="GPU not available"))])
def enable_gpu(request):
    return request.param