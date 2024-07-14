"""
Framework for Idealized Ocean Models (FRIDOM)
=============================================

Description
-----------
FRIDOM is a modeling framework designed with a singular goal in mind: 
to provide a high-level interface for the development of idealized ocean models. 
FRIDOM leverages the power of CUDA arrays on GPU through CuPy, enabling the 
execution of models at medium resolutions, constrained only by your hardware 
capabilities, right within Jupyter Notebook.

For more information, please visit the project's GitHub repository:
https://github.com/Gordi42/FRIDOM

Modules
-------
`framework`
    Contains the core classes and functions for the FRIDOM framework.
`nonhydro`
    A 3D non-hydrostatic model
`shallowwater`
    A 2D shallow water model
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from . import framework
    from . import nonhydro
    from . import shallowwater

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
# items in the all_imports_by_origin dictionary are imported as elements of a module
all_modules_by_origin = {
    "fridom": [
        "framework",
        "nonhydro",
        "shallowwater",
    ],
}

all_imports_by_origin = {}

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
