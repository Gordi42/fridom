"""
# Projection module for the nonhydrostatic model
The projection routines are used for flow decomposition. E.g. to
decompose the flow into geostrophic and ageostrophic components,
or to decompose the flow into balanced and unbalanced components.

## Linear Projections:
    - WaveSpectral          : Projection onto the linear wave mode
    - GeostrophicSpectral   : Projection onto the geostrophic mode
    - GeostrophicTimeAverage: Projection onto the geo. mode using a time average
    - DivergenceSpectral    : Projection onto the divergence mode

## Nonlinear Projections:
    - OptimalBalance        : Balancing using the optimal balance method
    - NNMD                  : Balancing using nonlinear normal mode decomposition
"""

from .divergence_spectral import DivergenceSpectral
from .wave_spectral import WaveSpectral
from fridom.framework.projection import GeostrophicTimeAverage

from .nnmd import NNMD
from fridom.framework.projection import OptimalBalance