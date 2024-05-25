from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.projection.projection import Projection

class GeostrophicSpectralBase(Projection):
    """
    Geostrophic projection using spectral discrete eigenvectors.
    """
    def __init__(self, grid:GridBase,
                 VecQ, VecP) -> None:
        """
        Constructor of the Projector using spectral eigenvectors.
        """
        super().__init__(grid)
        # Construct the eigenvectors
        self.q = VecQ(0, grid)
        self.p = VecP(0, grid)
        return

    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the geostrophic subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The geostrophic state.
        """
        return z.project(self.p, self.q)

# remove symbols from namespace
del GridBase, StateBase, Projection