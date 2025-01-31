"""Spectral grid module for the shallowwater model."""
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
base_sw = "fridom.shallowwater.grid.spectral"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base_sw}.grid": ["Grid"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
