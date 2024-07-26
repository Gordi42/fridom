import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules

    # import classes
    from .grid import Grid
    from .position import AxisOffset, Position
    from .fft import FFT
    from .finite_differences import FiniteDifferences
    from .linear_interpolation import LinearInterpolation
    from .polynomial_interpolation import PolynomialInterpolation
    

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
base = "fridom.framework.grid.cartesian"
all_modules_by_origin = {}

# items in the all_imports_by_origin dictionary are imported as elements of a module
all_imports_by_origin = { 
    f"{base}.grid": ["Grid"],
    f"{base}.position": ["AxisOffset", "Position"],
    f"{base}.fft": ["FFT"],
    f"{base}.finite_differences": ["FiniteDifferences"],
    f"{base}.linear_interpolation": ["LinearInterpolation"],
    f"{base}.polynomial_interpolation": ["PolynomialInterpolation"],
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
