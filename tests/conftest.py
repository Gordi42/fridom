import pytest
import fridom.framework as fr

# Fixture to enable GPU testing
# backends = [fr.config.Backend.NUMPY, fr.config.Backend.CUPY]
backends = [fr.config.Backend.JAX_GPU]
# backends = [fr.config.Backend.JAX_CPU]


@pytest.fixture(scope='module', params=backends)
def backend(request):
    fr.config.set_backend(request.param)
    return request.param