"""
Modules to calculate advection terms.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .centered_advection import CenteredAdvection
    from .second_order_advection import SecondOrderAdvection

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.advection"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.centered_advection": ["CenteredAdvection"],
    f"{base_path}.second_order_advection": ["SecondOrderAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
