# base projection class
from fridom.framework.projection.projection import Projection

# linear projections
from fridom.framework.projection \
    .wave_spectral import WaveSpectralBase
from fridom.framework.projection \
    .divergence_spectral import DivergenceSpectralBase
from fridom.framework.projection \
    .geostrophic_spectral import GeostrophicSpectralBase
from fridom.framework.projection \
    .geostrophic_time_average import GeostrophicTimeAverageBase

# nonlinear projections
from fridom.framework.projection \
    .nnmd import NNMDBase
from fridom.framework.projection \
    .optimal_balance import OptimalBalanceBase