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
    from .cg_pressure_solver import CGPressureSolver

# ================================================================
#  Setup lazy loading
# ================================================================

base_path = "fridom.nonhydro.modules.pressure_solvers"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.spectral_pressure_solver": ["SpectralPressureSolver"],
    f"{base_path}.cg_pressure_solver": ["CGPressureSolver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
