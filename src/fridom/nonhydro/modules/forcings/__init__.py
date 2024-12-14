"""Forcing modules for the nonhydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.modules.forcings import Relaxation

    from .gaussian_wave_maker import GaussianWaveMaker
    from .polarized_wave_maker import PolarizedWaveMaker

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.forcings"
fr_base_path = "fridom.framework.modules.forcings"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.gaussian_wave_maker": ["GaussianWaveMaker"],
    f"{base_path}.polarized_wave_maker": ["PolarizedWaveMaker"],
    f"{fr_base_path}.relaxation": ["Relaxation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
