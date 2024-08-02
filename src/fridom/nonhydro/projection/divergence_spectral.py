import fridom.framework as fr
import fridom.nonhydro as nh


class DivergenceSpectral(fr.projection.DivergenceSpectralBase):
    def __init__(self, 
                 mset: nh.ModelSettings,
                 VecQ: nh.State = nh.eigenvectors.VecQ,
                 VecP: nh.State = nh.eigenvectors.VecP,
                 use_discrete: bool = True) -> None:
        super().__init__(mset, VecQ, VecP, use_discrete)