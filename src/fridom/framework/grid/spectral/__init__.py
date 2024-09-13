
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules

    # import classes
    from .grid import Grid
    
# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid.spectral"

all_modules_by_origin = {
}

all_imports_by_origin = { 
    f"{base}.grid": ["Grid"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)