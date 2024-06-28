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
    Projection onto the inertia-gravity wave subspace using spectral eigenvectors.
    
    Parameters
    ----------
    `VecQ` : `State (constructor)`
        The constructor of an eigenvector.
    `VecP` : `State (constructor)`
        The constructor of the projection vector onto an eigenspace.
    
    Attributes
    ----------
    `q` : `State`
        The eigenvector of the inertia-gravity wave subspace.
    `p` : `State`
        The projection vector onto the inertia-gravity wave subspace.
    
    Methods
    -------
    `__call__(z: State) -> State`
        Project a state to the inertia-gravity wave subspace.
    """
    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP) -> None:
        super().__init__(mset)
        # use that the projection on the positive and negative eigenspaces
        # are the same. Hence, we only need to construct one of them.
        self.q = VecQ(1, mset)
        self.p = VecP(1, mset)
        return

    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the inertia-gravity wave subspace.
        
        Parameters
        ----------
        `z` : `State`
            The state to project.
        
        Returns
        -------
        `State`
            The projection of the state onto the inertia-gravity wave subspace.
        """
        return z.project(self.p, self.q) * 2