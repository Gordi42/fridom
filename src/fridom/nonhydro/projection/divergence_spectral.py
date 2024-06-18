from fridom.nonhydro.grid import Grid
from fridom.framework.projection \
    .divergence_spectral import DivergenceSpectralBase


class DivergenceSpectral(DivergenceSpectralBase):
    def __init__(self, grid: Grid) -> None:
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        super().__init__(grid, VecQ, VecP)

# remove symbols from namespace
del Grid, DivergenceSpectralBase