"""
Projection
===
The `projection` module provides classes for flow decomposition

Description
-----------
Projection classes are used to decompose the flow into different components.
For example, the flow can be decomposed into geostrophic and ageostrophic
components, or into balanced and unbalanced components.

Base Projection
---------------
`Projection`
    Base class for projections.

Linear Projections
------------------
`WaveSpectralBase`
    Projection onto the linear wave mode.
`DivergenceSpectralBase`
    Projection onto the divergence mode.
`GeostrophicSpectralBase`
    Projection onto the geostrophic mode.
`GeostrophicTimeAverage`
    Projection onto the geostrophic mode using a time average.

Nonlinear Projections
---------------------
`OptimalBalance`
    Balancing using the optimal balance method.
`NNMDBase`
    Balancing using nonlinear normal mode decomposition.
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # base projection class
    from .projection import Projection

    # linear projections
    from .wave_spectral import WaveSpectralBase
    from .divergence_spectral import DivergenceSpectralBase
    from .geostrophic_spectral import GeostrophicSpectralBase
    from .geostrophic_time_average import GeostrophicTimeAverage

    # nonlinear projections
    from .nnmd import NNMDBase
    from .optimal_balance import OptimalBalance

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
all_modules_by_origin = { }

# items in the all_imports_by_origin dictionary are imported as elements of a module
base_path = "fridom.framework.projection"
all_imports_by_origin = { 
    f"{base_path}.projection": ["Projection"],
    f"{base_path}.wave_spectral": ["WaveSpectralBase"],
    f"{base_path}.divergence_spectral": ["DivergenceSpectralBase"],
    f"{base_path}.geostrophic_spectral": ["GeostrophicSpectralBase"],
    f"{base_path}.geostrophic_time_average": ["GeostrophicTimeAverage"],
    f"{base_path}.nnmd": ["NNMDBase"],
    f"{base_path}.optimal_balance": ["OptimalBalance"],
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
