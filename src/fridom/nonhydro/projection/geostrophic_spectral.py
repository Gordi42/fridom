from fridom.framework.projection \
    .geostrophic_spectral import GeostrophicSpectralBase
from fridom.nonhydro.model_settings import ModelSettings

class GeostrophicSpectral(GeostrophicSpectralBase):
    def __init__(self, mset: ModelSettings) -> None:
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        super().__init__(mset, VecQ, VecP)