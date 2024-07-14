"""
Pressure Solvers
===
This module contains the pressure solvers for the non-hydrostatic model.

Classes
-------
`SpectralPressureSolver`
    The standard pressure solver for the non-hydrostatic model. (no topography)
`CGPressureSolver`
    The pressure solver for the non-hydrostatic model using the conjugate 
    gradient method. (with topography)
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .spectral_pressure_solver import SpectralPressureSolver
    from .cg_pressure_solver import CGPressureSolver

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
all_modules_by_origin = { }

# items in the all_imports_by_origin dictionary are imported as elements of a module
base_path = "fridom.nonhydro.modules.pressure_solvers"
all_imports_by_origin = {
    f"{base_path}.spectral_pressure_solver": ["SpectralPressureSolver"],
    f"{base_path}.cg_pressure_solver": ["CGPressureSolver"],
}

# ================================================================
#  Set up the import system
# ================================================================

origins = {}
_all_modules = []
for origin, items in all_modules_by_origin.items():
    for item in items:
        _all_modules.append(item)
        origins[item] = origin

_all_imports = []
for origin, items in all_imports_by_origin.items():
    for item in items:
        _all_imports.append(item)
        origins[item] = origin

# load submodules on demand
class _module(ModuleType):
    def __getattr__(self, name):
        # check if the attribute is a module
        if name in _all_modules:
            res = importlib.import_module(origins[name] + "." + name)
        # check if the attribute is an import
        elif name in _all_imports:
            mod = importlib.import_module(origins[name])
            res = getattr(mod, name)
        # if the attribute is not found
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        # set the attribute in the current module such that it is not loaded again
        setattr(self, name, res)
        # return the attribute
        return res

sys.modules[__name__].__class__ = _module
__all__ = _all_modules + _all_imports
