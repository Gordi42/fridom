from fridom.framework.projection \
    .nnmd import NNMDBase
from fridom.nonhydro.model_settings import ModelSettings

class NNMD(NNMDBase):
    def __init__(self, mset: ModelSettings, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.framework.model import Model
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        from fridom.nonhydro.state import State
        super().__init__(mset, Model, State, VecQ, VecP, order, enable_dealiasing)