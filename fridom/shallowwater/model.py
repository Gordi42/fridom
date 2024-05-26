from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State
from fridom.framework.model_base import ModelBase
from fridom.shallowwater.model_state import ModelState


class Model(ModelBase):
    """
    A scaled rotating shallow water model. The discretization is based on a
    energy conserving finite difference scheme on a staggered Arakawa C-grid.
    Based on Sadourny [1975].
    """

    def __init__(self, grid:Grid) -> None:
        """
        Constructor.

        Args:
            grid (Grid)             : Grid.
        """
        mset = grid.mset
        from fridom.shallowwater.state import State
        super().__init__(grid, 
                         State, 
                         ModelState,
                         is_spectral=False)
        self.mset = mset

        # Modules
        from fridom.shallowwater.modules.advection import SadournyAdvection
        from fridom.shallowwater.modules.linear_tendency import LinearTendency

        self.linear_tendency = LinearTendency()
        self.advection = SadournyAdvection()
        
        self.linear_tendency.start(grid=grid, timer=self.timer)
        self.advection.start(grid=grid, timer=self.timer)
        return


    # ============================================================
    #   TOTAL TENDENCY
    # ============================================================

    def total_tendency(self) -> None:
        """
        Calculate total tendency. (Righthand side of the PDE)
        """

        self.linear_tendency.update(self.model_state, self.dz)
        if self.mset.enable_nonlinear:
            self.advection.update(self.model_state, self.dz)

        return



# remove symbols from namespace
del Grid, State, ModelBase