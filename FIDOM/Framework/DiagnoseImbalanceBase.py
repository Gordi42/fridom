from FIDOM.Framework.ModelSettingsBase import ModelSettingsBase
from FIDOM.Framework.GridBase import GridBase
from FIDOM.Framework.StateBase import StateBase
from FIDOM.Framework.ModelBase import ModelBase
from FIDOM.Framework.ProjectionBase import Projection

class DiagnoseImbalanceBase:
    def __init__(self, mset:ModelSettingsBase, grid:GridBase,
                 Model:ModelBase,
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
        self.model = Model(mset, grid)

        # prepare results
        self.z_ini = None
        self.z_ini_bal = None
        self.z_fin = None
        self.z_fin_bal = None
        self.imbalance = None

    def __call__(self, z:StateBase) -> float:
        verbose = self.mset.print_verbose
        model = self.model
        model.reset()

        self.z_ini = z.copy() if self.store_details else None
        
        verbose("Running initial projection")
        z_bal = self.proj_ini(z)

        self.z_ini_bal = z_bal.copy() if self.store_details else None

        verbose(f"Running model for {self.diag_per} seconds")
        model.z = z_bal
        model.run(runlen=self.diag_per)

        self.z_fin = model.z.copy() if self.store_details else None

        verbose("Running final projection")
        z_bal = self.proj_fin(model.z)

        self.z_fin_bal = z_bal.copy() if self.store_details else None

        verbose("Calculating imbalance")
        self.imbalance = z_bal.norm_of_diff(model.z)
        return self.imbalance
