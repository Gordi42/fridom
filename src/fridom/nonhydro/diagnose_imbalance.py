from fridom.framework.projection.projection import Projection
from fridom.framework.diagnose_imbalance_base import DiagnoseImbalanceBase
from fridom.nonhydro.model_settings import ModelSettings


class DiagnoseImbalance(DiagnoseImbalanceBase):
    def __init__(self, mset: ModelSettings,
                 diag_per:float,
                 proj:Projection,
                 proj2=None,
                 store_details=False) -> None:
        from fridom.framework.model import Model
        super().__init__(mset, Model, diag_per, proj, proj2, store_details)