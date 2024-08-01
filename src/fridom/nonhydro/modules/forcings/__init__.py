"""
This module contains classes that generate forcing terms for the nonhydrostatic model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .gaussian_wave_maker import GaussianWaveMaker
    from .polarized_wave_maker import PolarizedWaveMaker

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.forcings"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.gaussian_wave_maker": ["GaussianWaveMaker"],
    f"{base_path}.polarized_wave_maker": ["PolarizedWaveMaker"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
