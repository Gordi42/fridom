# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.projection.projection import Projection
# Import type information
if TYPE_CHECKING:
    from fridom.framework.modelsettings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase


class DivergenceSpectralBase(Projection):
    """
    Projection onto the divergence subspace using spectral discrete eigenvectors.
    """

    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP) -> None:
        """
        Constructor of the Projector using spectral eigenvectors.
        """
        super().__init__(mset)
        self.q = VecQ("d", mset)
        self.p = VecP("d", mset)

    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the divergence subspace.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The divergent state.
        """
        return z.project(self.p, self.q)