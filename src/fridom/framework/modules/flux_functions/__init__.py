"""Flux functions module."""

from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .flux_function_base import FluxFunctionBase
    from .upwind import Upwind

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.modules.flux_functions"

all_modules_by_origin = {
}

all_imports_by_origin = {
    f"{base}.flux_function_base": ["FluxFunctionBase"],
    f"{base}.upwind": ["Upwind"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
