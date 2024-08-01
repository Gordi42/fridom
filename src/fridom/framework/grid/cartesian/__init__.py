from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    from . import fft

    # import classes
    from .grid import Grid
    from .position import AxisOffset, Position
    from .fft import FFT
    from .finite_differences import FiniteDifferences
    from .linear_interpolation import LinearInterpolation
    from .polynomial_interpolation import PolynomialInterpolation
    
# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid.cartesian"

all_modules_by_origin = {
    base: ["fft"],
}

all_imports_by_origin = { 
    f"{base}.grid": ["Grid"],
    f"{base}.position": ["AxisOffset", "Position"],
    f"{base}.fft": ["FFT"],
    f"{base}.finite_differences": ["FiniteDifferences"],
    f"{base}.linear_interpolation": ["LinearInterpolation"],
    f"{base}.polynomial_interpolation": ["PolynomialInterpolation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)