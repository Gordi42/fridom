from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules
    from . import eigenvectors

    # import classes
    from .grid import Grid
    from fridom.framework.grid.cartesian.fft import FFT
    from fridom.framework.grid.cartesian.finite_differences import FiniteDifferences
    from fridom.framework.grid.cartesian.linear_interpolation import LinearInterpolation
    from fridom.framework.grid.cartesian.polynomial_interpolation import PolynomialInterpolation
    
# ================================================================
#  Setup lazy loading
# ================================================================
base_fr = "fridom.framework.grid.cartesian"
base_nh = "fridom.nonhydro.grid.cartesian"

all_modules_by_origin = {
    base_nh: ["eigenvectors"],
}

all_imports_by_origin = { 
    f"{base_nh}.grid": ["Grid"],
    f"{base_fr}.fft": ["FFT"],
    f"{base_fr}.finite_differences": ["FiniteDifferences"],
    f"{base_fr}.linear_interpolation": ["LinearInterpolation"],
    f"{base_fr}.polynomial_interpolation": ["PolynomialInterpolation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
