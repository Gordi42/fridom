"""
# Diffusion module
This module contains the diffusion modules (friction and mixing) of the model.

## Available modules:
- HarmonicFriction: Harmonic friction module.
- HarmonicMixing: Harmonic mixing module.
- BiharmonicFriction: Biharmonic friction module.
- BiharmonicMixing: Biharmonic mixing module.
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .harmonic_friction import HarmonicFriction
    from .harmonic_mixing import HarmonicMixing
    from .biharmonic_friction import BiharmonicFriction
    from .biharmonic_mixing import BiharmonicMixing

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
all_modules_by_origin = { }

# items in the all_imports_by_origin dictionary are imported as elements of a module
base_path = "fridom.nonhydro.modules.diffusion"
all_imports_by_origin = { 
    f"{base_path}.harmonic_friction": ["HarmonicFriction"],
    f"{base_path}.harmonic_mixing": ["HarmonicMixing"],
    f"{base_path}.biharmonic_friction": ["BiharmonicFriction"],
    f"{base_path}.biharmonic_mixing": ["BiharmonicMixing"],
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
