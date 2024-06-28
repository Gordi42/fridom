from fridom.framework.projection.wave_spectral import WaveSpectralBase
from fridom.nonhydro.model_settings import ModelSettings

class WaveSpectral(WaveSpectralBase):
    def __init__(self, mset: ModelSettings) -> None:
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        super().__init__(mset, VecQ, VecP)