"""Framework for Idealized Ocean Models (FRIDOM).

==============================================

Description
-----------
FRIDOM is a modeling framework designed with a singular goal in mind:
to provide a high-level interface for the development of idealized ocean models.
FRIDOM leverages the power of CUDA arrays on GPU through CuPy, enabling the
execution of models at medium resolutions, constrained only by your hardware
capabilities, right within Jupyter Notebook.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/FRIDOM
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # isort: off
    from . import framework
    from . import nonhydro
    from . import shallowwater

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom": [
        "framework",
        "nonhydro",
        "shallowwater",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
