from fridom.framework.projection \
    .divergence_spectral import DivergenceSpectralBase
from fridom.nonhydro.model_settings import ModelSettings


class DivergenceSpectral(DivergenceSpectralBase):
    def __init__(self, mset: ModelSettings) -> None:
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        super().__init__(mset, VecQ, VecP)