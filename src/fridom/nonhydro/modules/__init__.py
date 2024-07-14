"""
Nonhydrostatic Model Modules
============================
A collection of modules for the nonhydrostatic model.

Modules
-------
`animation`
    Modules for creating animated output of the model.
`advection`
    Modules for advection schemes.
`interpolation`
    Modules for interpolation schemes.
`diffusion`
    Modules for diffusion schemes.
`pressure_solvers`
    Modules for pressure solver schemes.
`forcings`
    Modules for forcing schemes.

Classes
-------
`RestartModule`
    A module for restarting the model.
`BoundaryConditions`
    Boundary conditions module.
`LinearTendency`
    Linear tendency term.
`TendencyDivergence`
    Tendency divergence term (required for pressure solver).
`PressureGradientTendency`
    Pressure gradient tendency term.
`MainTendency`
    Main tendency module.
`NetCDFWriter`
    NetCDF writer module (for parallel output).
`NetCDFWriterSingleProc`
    NetCDF writer module for single processor.
`Diagnostics`
    Diagnostic module.

"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # importing modules
    from . import advection
    from . import interpolation
    from . import diffusion
    from . import pressure_solvers
    from . import forcings

    # importing classes
    from .boundary_conditions import BoundaryConditions
    from .linear_tendency import LinearTendency
    from .tendency_divergence import TendencyDivergence
    from .pressure_gradient_tendency import PressureGradientTendency
    from .main_tendency import MainTendency
    from .netcdf_writer_single_proc import NetCDFWriterSingleProc
    from .diagnostics import Diagnostics

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework.modules import animation

    # importing classes
    from fridom.framework.modules.netcdf_writer import NetCDFWriter
    from fridom.framework.modules.restart_module import RestartModule

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
base_path = "fridom.nonhydro.modules"
fr_base_path = "fridom.framework.modules"
all_modules_by_origin = { 
    base_path: [
        "advection", 
        "interpolation", 
        "diffusion", 
        "pressure_solvers", 
        "forcings"], 
    fr_base_path: ["animation"],
}

# items in the all_imports_by_origin dictionary are imported as elements of a module
all_imports_by_origin = { 
    f"{base_path}.boundary_conditions": ["BoundaryConditions"],
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{base_path}.tendency_divergence": ["TendencyDivergence"],
    f"{base_path}.pressure_gradient_tendency": ["PressureGradientTendency"],
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{base_path}.netcdf_writer_single_proc": ["NetCDFWriterSingleProc"],
    f"{base_path}.diagnostics": ["Diagnostics"],
    f"{fr_base_path}.netcdf_writer": ["NetCDFWriter"],
    f"{fr_base_path}.restart_module": ["RestartModule"], 
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
