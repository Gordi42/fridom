"""A collection of modules for the shallowwater model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.modules.advection import AdvectionBase

    from .sadourny_advection import SadournyAdvection

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.shallowwater.modules.advection"
fr_base_path = "fridom.framework.modules.advection"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{fr_base_path}": ["AdvectionBase"],
    f"{base_path}.sadourny_advection": ["SadournyAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
