"""
Advection modules for 1D, 2D, and 3D fluid simulations.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import classes
    from .advection_base import AdvectionBase
    from .centered_advection import CenteredAdvection

    # import modules


# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.framework.modules.advection"
all_imports_by_origin = {
    f"{base}.advection_base": ["AdvectionBase"],
    f"{base}.centered_advection": ["CenteredAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)