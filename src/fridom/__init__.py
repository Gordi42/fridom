"""
Framework for Idealized Ocean Models (FRIDOM).

==============================================

Description
-----------
FRIDOM is a modeling framework designed with a singular goal in mind:
to provide a high-level interface for the development of idealized
ocean models.
FRIDOM is built on JAX, enabling jit-compiled model execution on CPUs,
GPUs, and TPUs at medium resolutions, constrained only by your hardware
capabilities, right within Jupyter Notebook.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/FRIDOM
"""
from typing import TYPE_CHECKING

from lazypimp import setup

from fridom._compile_cache import configure as _configure_compile_cache

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from . import (
        benchmarking,
        framework,
        hydrostatic,
        model,
        nonhydro,
        shallowwater,
        spatial,
    )

    # root alias: ``fr.io`` re-exports ``fridom.model.io``
    from .model import io

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom": [
        "benchmarking",
        "framework",
        "hydrostatic",
        "model",
        "nonhydro",
        "shallowwater",
        "spatial",
    ],
    # root alias: ``fr.io`` resolves to ``fridom.model.io``
    "fridom.model": ["io"],
}

all_imports_by_origin = {}

# Enable the persistent JAX compilation cache before any lazy import can
# trigger a compile (see fridom/_compile_cache.py); a no-op when already
# configured or disabled via FRIDOM_DISABLE_COMPILE_CACHE.
_configure_compile_cache()

setup(__name__, all_modules_by_origin, all_imports_by_origin)
