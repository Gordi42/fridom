"""
All grid related classes and functions.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    from . import cartesian

    # import classes
    from .grid_base import GridBase
    from .diff_base import DiffBase
    from .interpolation_base import InterpolationBase
    from .position import AxisPosition, Position
    from .transform_type import TransformType
    
# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid"

all_modules_by_origin = { base: ["cartesian"] }

all_imports_by_origin = { 
    f"{base}.grid_base": ["GridBase"],
    f"{base}.diff_base": ["DiffBase"],
    f"{base}.interpolation_base": ["InterpolationBase"],
    f"{base}.position": ["AxisPosition", "Position"],
    f"{base}.transform_type": ["TransformType"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)