"""Base module for spectral grid classes."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
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
