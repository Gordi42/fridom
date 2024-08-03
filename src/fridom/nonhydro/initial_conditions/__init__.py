"""
Initial Conditions for the Nonhydrostatic Model
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .single_wave import SingleWave
    from .kelvin_wave import KelvinWave
    from .wave_package import WavePackage
    from .barotropic_jet import BarotropicJet
    from .jet import Jet
    from .coherent_eddy import CoherentEddy
    from .geostrophic_spectra import geostrophic_energy_spectrum, \
                                     RandomGeostrophicSpectra
    
# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.initial_conditions"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.single_wave": ["SingleWave"],
    f"{base_path}.kelvin_wave": ["KelvinWave"],
    f"{base_path}.wave_package": ["WavePackage"],
    f"{base_path}.barotropic_jet": ["BarotropicJet"],
    f"{base_path}.jet": ["Jet"],
    f"{base_path}.coherent_eddy": ["CoherentEddy"],
    f"{base_path}.geostrophic_spectra": ["geostrophic_energy_spectrum", 
                                         "RandomGeostrophicSpectra"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
