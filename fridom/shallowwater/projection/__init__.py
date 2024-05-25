"""
# Projection module for the shallow water model
The projection routines are used for flow decomposition. E.g. to
decompose the flow into geostrophic and ageostrophic components,
or to decompose the flow into balanced and unbalanced components.

## Linear Projections:
    - WaveSpectral          : Projection onto the linear wave mode
    - GeostrophicSpectral   : Projection onto the geostrophic mode
    - GeostrophicTimeAverage: Projection onto the geo. mode using a time average

## Nonlinear Projections:
    - OptimalBalance        : Balancing using the optimal balance method
    - NNMD                  : Balancing using nonlinear normal mode decomposition
"""

# linear projections
from .wave_spectral import WaveSpectral
from .geostrophic_spectral import GeostrophicSpectral
from .geostrophic_time_average import GeostrophicTimeAverage

# nonlinear projections
from .optimal_balance import OptimalBalance
from .nnmd import NNMD