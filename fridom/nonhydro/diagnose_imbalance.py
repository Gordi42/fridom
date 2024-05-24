from fridom.nonhydro.grid import Grid
from fridom.framework.projection_base import Projection
from fridom.framework.diagnose_imbalance_base import DiagnoseImbalanceBase


class DiagnoseImbalance(DiagnoseImbalanceBase):
    def __init__(self, grid:Grid,
                 diag_per:float,
                 proj:Projection,
                 proj2=None,
                 store_details=False) -> None:
        from fridom.nonhydro.model import Model
        super().__init__(grid, Model, diag_per, proj, proj2, store_details)

# remove symbols from namespace
del Grid, Projection, DiagnoseImbalanceBase