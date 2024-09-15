"""
Forcing modules. E.g. relaxation
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import classes
    from .relaxation import Relaxation

    # import modules


# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.framework.modules.forcings"
all_imports_by_origin = {
    f"{base}.relaxation": ["Relaxation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)