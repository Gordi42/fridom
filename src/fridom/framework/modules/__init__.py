"""
Modules
=======
Base classes and functions for creating modules in the model.
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
    from . import animation

    # importing the Classes
    from .module import Module
    from .module_container import ModuleContainer
    from .restart_module import RestartModule
    from .boundary_conditions import BoundaryConditions
    from .netcdf_writer_single_proc import NetCDFWriterSingleProc
    from .netcdf_writer import NetCDFWriter

    # importing the functions
    from .module import setup_module, module_method
    

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
base_path = "fridom.framework.modules"
all_modules_by_origin = { base_path: ["animation"] }

# items in the all_imports_by_origin dictionary are imported as elements of a module
all_imports_by_origin = { 
    f"{base_path}.module": ["Module", "setup_module", "module_method"],
    f"{base_path}.module_container": ["ModuleContainer"],
    f"{base_path}.restart_module": ["RestartModule"],
    f"{base_path}.boundary_conditions": ["BoundaryConditions"],
    f"{base_path}.netcdf_writer_single_proc": ["NetCDFWriterSingleProc"],
    f"{base_path}.netcdf_writer": ["NetCDFWriter"],
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
