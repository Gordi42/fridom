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

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.advection"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.centered_advection": ["CenteredAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
