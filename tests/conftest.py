"""Configuration for pytest."""
import logging
from io import StringIO
import os
import sys

import pytest

import fridom.framework as fr

# Get the backend from the environment variable.
backend = os.getenv("FRIDOM_BACKEND", None)

# check if the backend is the same as the one in the config
if backend is not None and fr.config.backend != backend:
    sys.exit(f"Backend {backend} is not available")

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
