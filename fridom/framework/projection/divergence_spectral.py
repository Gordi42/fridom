from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.projection.projection import Projection


class DivergenceSpectralBase(Projection):
    """
    Projection onto the divergence subspace using spectral discrete eigenvectors.
    """

    def __init__(self, grid: GridBase,
                 VecQ, VecP) -> None:
        """
        Constructor of the Projector using spectral eigenvectors.
        """
        super().__init__(grid)
        self.q = VecQ("d", grid)
        self.p = VecP("d", grid)

    def __call__(self, z: StateBase) -> StateBase:
        """
        Project a state to the divergence subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The divergent state.
        """
        return z.project(self.p, self.q)

# remove symbols from namespace
del GridBase, StateBase, Projection