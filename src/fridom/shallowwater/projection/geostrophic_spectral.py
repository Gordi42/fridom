from fridom.shallowwater.grid import Grid
from fridom.framework.projection \
    .geostrophic_spectral import GeostrophicSpectralBase

class GeostrophicSpectral(GeostrophicSpectralBase):
    def __init__(self, grid: Grid) -> None:
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        super().__init__(grid, VecQ, VecP)

# remove symbols from the namespace
del Grid, GeostrophicSpectralBase