"""Configuration for pytest."""
import os
import sys

import fridom.framework as fr

# Get the backend from the environment variable.
backend = os.getenv("FRIDOM_BACKEND", None)

# check if the backend is the same as the one in the config
if backend is not None and fr.config.backend != backend:
    sys.exit(f"Backend {backend} is not available")
