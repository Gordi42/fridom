"""A collection of modules for the nonhydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.modules.advection import AdvectionBase, CenteredAdvection

    from .spectral_advection import SpectralAdvection

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.advection"
fr_base_path = "fridom.framework.modules.advection"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{fr_base_path}": ["AdvectionBase", "CenteredAdvection"],
    f"{base_path}.spectral_advection": ["SpectralAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
