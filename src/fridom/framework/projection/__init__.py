"""
Projection
===
The `projection` module provides classes for flow decomposition

Description
-----------
Projection classes are used to decompose the flow into different components.
For example, the flow can be decomposed into geostrophic and ageostrophic
components, or into balanced and unbalanced components.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # base projection class
    from .projection import Projection

    # linear projections
    from .spectral_projections import GeostrophicSpectral, WaveSpectral, DivergenceSpectral
    from .geostrophic_time_average import GeostrophicTimeAverage

    # nonlinear projections
    from .optimal_balance import OptimalBalance
    from .nnmd import NNMD

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.projection"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.projection": ["Projection"],
    f"{base_path}.spectral_projections": [
        "GeostrophicSpectral", "WaveSpectral", "DivergenceSpectral"],
    f"{base_path}.geostrophic_time_average": ["GeostrophicTimeAverage"],
    f"{base_path}.optimal_balance": ["OptimalBalance"],
    f"{base_path}.nnmd": ["NNMD"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
