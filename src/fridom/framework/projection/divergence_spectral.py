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
    Projection onto the divergence subspace using spectral eigenvectors.
    
    Parameters
    ----------
    `VecQ` : `State (constructor)`
        The constructor of an eigenvector.
    `VecP` : `State (constructor)`
        The constructor of the projection vector onto an eigenspace.
    
    Attributes
    ----------
    `q` : `State`
        The eigenvector of the divergent subspace.
    `p` : `State`
        The projection vector onto the divergent subspace.
    
    Methods
    -------
    `__call__(z: State) -> State`
        Project a state to the divergence subspace.
    """
    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP) -> None:
        super().__init__(mset)
        self.q = VecQ("d", mset)
        self.p = VecP("d", mset)
        return

    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the divergent subspace.
        
        Parameters
        ----------
        `z` : `State`
            The state to project.
        
        Returns
        -------
        `State`
            The projection of the state onto the divergent subspace.
        """
        return z.project(self.p, self.q)