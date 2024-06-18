from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.projection.projection import Projection


class WaveSpectralBase(Projection):
    """
    Inertia-gravity wave projection using spectral discrete eigenvectors.
    """
    def __init__(self, grid:GridBase,
                 VecQ, VecP) -> None:
        """
        Constructor of the Projector using spectral eigenvectors.
        """
        super().__init__(grid)
        # use that the projection on the positive and negative eigenspaces
        # are the same. Hence, we only need to construct one of them.
        self.q = VecQ(1, grid)
        self.p = VecP(1, grid)
        return

    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the inertia-gravity wave subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The wave mode state.
        """
        return z.project(self.p, self.q) * 2

# remove symbols from namespace
del GridBase, StateBase, Projection