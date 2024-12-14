"""All grid related classes and functions."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import cartesian, spectral
    from .boundary_type import BCType
    from .diff_module import DiffModule
    from .dummy_interpolation import DummyInterpolation
    from .fft_padding import FFTPadding

    # import all classes
    from .grid_base import GridBase
    from .interpolation_module import InterpolationModule
    from .position import AxisPosition, Position
    from .water_mask import WaterMask

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid"

all_modules_by_origin = { base: ["cartesian",
                                 "spectral"] }

all_imports_by_origin = {
    f"{base}.grid_base": ["GridBase"],
    f"{base}.diff_module": ["DiffModule"],
    f"{base}.interpolation_module": ["InterpolationModule"],
    f"{base}.dummy_interpolation": ["DummyInterpolation"],
    f"{base}.position": ["AxisPosition", "Position"],
    f"{base}.water_mask": ["WaterMask"],
    f"{base}.boundary_type": ["BCType"],
    f"{base}.fft_padding": ["FFTPadding"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
