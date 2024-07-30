"""
Initial Conditions for the Nonhydrostatic Model
===============================================

Classes
-------
`SingleWave`
    A single polarized sine wave in the whole domain
`WavePackage`
    A polarized wave package with a Gaussian profile
`VerticalMode`
    A polarized vertical mode over the whole domain
`BarotropicJet`
    A barotropic unstable jet with horizontal shear
`Jet`
    An unstable jet with horizontal and vertical shear
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .single_wave import SingleWave
    from .kelvin_wave import KelvinWave
    from .wave_package import WavePackage
    from .vertical_mode import VerticalMode
    from .barotropic_jet import BarotropicJet
    from .jet import Jet
    from .geostrophic_spectra import geostrophic_energy_spectrum, \
                                     RandomGeostrophicSpectra
    

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
all_modules_by_origin = { }

# items in the all_imports_by_origin dictionary are imported as elements of a module
base_path = "fridom.nonhydro.initial_conditions"
all_imports_by_origin = { 
    f"{base_path}.single_wave": ["SingleWave"],
    f"{base_path}.kelvin_wave": ["KelvinWave"],
    f"{base_path}.wave_package": ["WavePackage"],
    f"{base_path}.vertical_mode": ["VerticalMode"],
    f"{base_path}.barotropic_jet": ["BarotropicJet"],
    f"{base_path}.jet": ["Jet"],
    f"{base_path}.geostrophic_spectra": ["geostrophic_energy_spectrum", 
                                         "RandomGeostrophicSpectra"],
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
