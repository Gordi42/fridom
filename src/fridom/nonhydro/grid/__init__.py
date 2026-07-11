"""Module that stores all available grids for the non-hydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.grid import AxisPosition, BCType, Position

    from . import cartesian

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.nonhydro.grid"

all_modules_by_origin = { base: ["cartesian"] }

all_imports_by_origin = {
    "fridom.framework.grid": ["AxisPosition", "Position", "BCType"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
