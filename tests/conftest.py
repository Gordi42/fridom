"""Configuration for pytest."""
import logging
from io import StringIO

import jax
import pytest

import fridom.framework as fr


# ================================================================
#  Device-count markers
# ================================================================
def pytest_runtest_setup(item):
    """Skip tests whose device-count requirements are not met."""
    n_devices = jax.device_count()
    if item.get_closest_marker("multi_device") and n_devices == 1:
        pytest.skip("requires multiple jax devices")
    if item.get_closest_marker("single_device") and n_devices > 1:
        pytest.skip("requires a single jax device")


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def capture_logs():
    """Fixture to capture log output."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    fr.log.addHandler(handler)
    yield stream
    fr.log.removeHandler(handler)
    fr.log.setLevel("SILENT")
