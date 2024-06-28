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