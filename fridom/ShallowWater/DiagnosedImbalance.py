from fridom.ShallowWater.Grid import Grid
from fridom.Framework.ProjectionBase import Projection
from fridom.Framework.DiagnoseImbalanceBase import DiagnoseImbalanceBase


class DiagnoseImbalance(DiagnoseImbalanceBase):
    def __init__(self, grid:Grid,
                 diag_per:float,
                 proj:Projection,
                 proj2=None,
                 store_details=False) -> None:
        from fridom.ShallowWater.Model import Model
        super().__init__(grid, Model, diag_per, proj, proj2, store_details)

# remove symbols from namespace
del Grid, Projection, DiagnoseImbalanceBase