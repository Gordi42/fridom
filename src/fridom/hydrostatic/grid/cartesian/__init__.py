"""Cartesian grid classes for the hydrosatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.grid.cartesian.fft import FFT
    from fridom.framework.grid.cartesian.finite_differences import FiniteDifferences
    from fridom.framework.grid.cartesian.linear_interpolation import LinearInterpolation
    from fridom.framework.grid.cartesian.polynomial_interpolation import (
        PolynomialInterpolation,
    )

    from .grid import Grid

# ================================================================
#  Setup lazy loading
# ================================================================
base_fr = "fridom.framework.grid.cartesian"
base_hs = "fridom.hydrostatic.grid.cartesian"

all_modules_by_origin = {
}

all_imports_by_origin = {
    f"{base_hs}.grid": ["Grid"],
    f"{base_fr}.fft": ["FFT"],
    f"{base_fr}.finite_differences": ["FiniteDifferences"],
    f"{base_fr}.linear_interpolation": ["LinearInterpolation"],
    f"{base_fr}.polynomial_interpolation": ["PolynomialInterpolation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
