"""
Projection module for the nonhydrostatic model

The projection routines are used for flow decomposition. E.g. to
decompose the flow into geostrophic and ageostrophic components,
or to decompose the flow into balanced and unbalanced components.

"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .divergence_spectral import DivergenceSpectral
    from .geostrophic_spectral import GeostrophicSpectral
    from .wave_spectral import WaveSpectral
    from fridom.framework.projection import GeostrophicTimeAverage

    from .nnmd import NNMD
    from fridom.framework.projection import OptimalBalance

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.projection"
fr_base_path = "fridom.framework.projection"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.wave_spectral": ["WaveSpectral"],
    f"{base_path}.geostrophic_spectral": ["GeostrophicSpectral"],
    f"{fr_base_path}.geostrophic_time_average": ["GeostrophicTimeAverage"],
    f"{base_path}.divergence_spectral": ["DivergenceSpectral"],
    f"{fr_base_path}.optimal_balance": ["OptimalBalance"],
    f"{base_path}.nnmd": ["NNMD"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
