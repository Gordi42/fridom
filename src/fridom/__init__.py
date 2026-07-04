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

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from . import benchmarking, framework, nonhydro, shallowwater

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom": [
        "benchmarking",
        "framework",
        "nonhydro",
        "shallowwater",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
