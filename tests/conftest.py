"""Configuration for pytest."""
import logging
import os
from io import StringIO
from pathlib import Path

# On a single GPU, jax preallocates ~75% of the device memory per
# process (see XLA_PYTHON_CLIENT_PREALLOCATE). With pytest-xdist that
# lets only one worker fit and the rest stall, so disable preallocation
# and allocate on demand instead. Must be set before jax initializes
# its backend; setdefault keeps an explicit user override. Harmless on
# cpu (no-op).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import pytest

import fridom.framework as fr

# ================================================================
#  Persistent compilation cache
# ================================================================
# The suite is dominated by many small jit compilations. A persistent,
# on-disk cache turns these into cache hits, roughly halving warm-run
# wall time on both cpu and gpu. The cache is keyed on the HLO, jaxlib
# version, and backend, so it stays correct across code changes.
# Override the location with FRIDOM_TEST_JAX_CACHE_DIR; set it empty to
# disable caching.
_jax_cache_dir = os.environ.get(
    "FRIDOM_TEST_JAX_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".jax_cache"))
# Give each pytest-xdist worker its own cache subdirectory. jax's local
# cache is not written atomically, so a shared directory lets one worker
# read a half-written entry (a truncated-zlib error that, under the
# filterwarnings=error policy, fails the test). Per-worker directories
# avoid the race entirely; warm reruns still hit because --dist loadfile
# keeps the file->worker assignment stable.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _jax_cache_dir and _worker:
    _jax_cache_dir = str(Path(_jax_cache_dir) / _worker)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir or None)
# the defaults skip sub-second / small compilations, which is exactly
# what the test suite consists of; cache everything instead
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


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
