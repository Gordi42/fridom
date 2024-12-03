"""Configuration for pytest."""
import os
import sys
import fridom.framework as fr

# Get the backend from the environment variable.
backend = os.getenv("FRIDOM_BACKEND", "numpy")

# check if the backend is the same as the one in the config
if fr.config.backend != backend:
    sys.exit(f"Backend {backend} is not available")

