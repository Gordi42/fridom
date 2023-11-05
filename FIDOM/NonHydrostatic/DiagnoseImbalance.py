from FIDOM.NonHydrostatic.ModelSettings import ModelSettings
from FIDOM.NonHydrostatic.Grid import Grid
from FIDOM.Framework.ProjectionBase import Projection
from FIDOM.Framework.DiagnoseImbalanceBase import DiagnoseImbalanceBase


class DiagnoseImbalance(DiagnoseImbalanceBase):
    def __init__(self, mset:ModelSettings, grid:Grid,
                 diag_per:float,
                 proj:Projection,
                 proj2=None,
                 store_details=False) -> None:
        """
        Calculate the diagnostic imbalance

        Arguments:
            mset (ModelSettings) : Model settings
            grid (Grid)          : Grid
            diag_per (float)     : Model run time in between the two projections
            proj (Projection)    : Projector to be tested
            proj2 (Projection)   : Second projector for cross balancing
                                   if None, then proj is used
            store_details (bool) : Whether to store the fields
        """

        self.mset = mset
        self.grid = grid
        self.diag_per = diag_per
        self.proj_ini = proj
        self.proj_fin = proj2 if proj2 is not None else proj
        self.store_details = store_details
        from FIDOM.NonHydrostatic.Model import Model
        self.model = Model(mset, grid)

        # prepare results
        self.z_ini = None
        self.z_ini_bal = None
        self.z_fin = None
        self.z_fin_bal = None
        self.imbalance = None