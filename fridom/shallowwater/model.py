from fridom.shallowwater.grid import Grid
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
        from fridom.shallowwater.state import State
        super().__init__(grid, State, ModelState, is_spectral=False)
        self.mset = grid.mset
        return

# remove symbols from namespace
del Grid, ModelBase