"""
This module contains the pressure solvers for the non-hydrostatic model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .spectral_pressure_solver import SpectralPressureSolver

# ================================================================
#  Setup lazy loading
# ================================================================

base_path = "fridom.nonhydro.modules.pressure_solvers"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.spectral_pressure_solver": ["SpectralPressureSolver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
