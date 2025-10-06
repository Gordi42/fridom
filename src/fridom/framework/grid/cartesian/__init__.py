"""Base module for Cartesian grids."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import discrete_spectral_operators

    # import all classes
    from .eno_interpolation import InterENO
    from .fft import FFT
    from .finite_differences import FiniteDifferences
    from .grid import Grid
    from .linear_interpolation import LinearInterpolation
    from .polynomial_interpolation import PolynomialInterpolation
    from .spectral_diff import SpectralDiff
    from .weno_interpolation import InterWENO

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid.cartesian"

all_modules_by_origin = {
    base: ["discrete_spectral_operators"],
}

all_imports_by_origin = {
    f"{base}.grid": ["Grid"],
    f"{base}.eno_interpolation": ["InterENO"],
    f"{base}.fft": ["FFT"],
    f"{base}.finite_differences": ["FiniteDifferences"],
    f"{base}.spectral_diff": ["SpectralDiff"],
    f"{base}.linear_interpolation": ["LinearInterpolation"],
    f"{base}.polynomial_interpolation": ["PolynomialInterpolation"],
    f"{base}.weno_interpolation": ["InterWENO"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
