# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.projection.projection import Projection
# Import type information
if TYPE_CHECKING:
    from fridom.framework.modelsettings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase


class WaveSpectralBase(Projection):
    """
    Inertia-gravity wave projection using spectral discrete eigenvectors.
    """
    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP) -> None:
        """
        Constructor of the Projector using spectral eigenvectors.
        """
        super().__init__(mset)
        # use that the projection on the positive and negative eigenspaces
        # are the same. Hence, we only need to construct one of them.
        self.q = VecQ(1, mset)
        self.p = VecP(1, mset)
        return

    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the inertia-gravity wave subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The wave mode state.
        """
        return z.project(self.p, self.q) * 2