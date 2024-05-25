"""
# Base Classes for Projections
The projection routines are used for flow decomposition. E.g. to
decompose the flow into geostrophic and ageostrophic components,
or to decompose the flow into balanced and unbalanced components.

## Base Projections:
    - Projection            : Base class for projections

## Linear Projections:
    - WaveSpectral          : Projection onto the linear wave mode
    - GeostrophicSpectral   : Projection onto the geostrophic mode
    - GeostrophicTimeAverage: Projection onto the geo. mode using a time average
    - DivergenceSpectral    : Projection onto the divergence mode

## Nonlinear Projections:
    - OptimalBalance        : Balancing using the optimal balance method
    - NNMD                  : Balancing using nonlinear normal mode decomposition
"""

# base projection class
from .projection import Projection

# linear projections
from .wave_spectral import WaveSpectralBase
from .divergence_spectral import DivergenceSpectralBase
from .geostrophic_spectral import GeostrophicSpectralBase
from .geostrophic_time_average import GeostrophicTimeAverageBase

# nonlinear projections
from .nnmd import NNMDBase
from .optimal_balance import OptimalBalanceBase