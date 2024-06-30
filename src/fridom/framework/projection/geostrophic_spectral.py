# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.projection.projection import Projection
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase

class GeostrophicSpectralBase(Projection):
    """
    Projection onto the geostrophic subspace using spectral eigenvectors.

    Parameters
    ----------
    `VecQ` : `State (constructor)`
        The constructor of an eigenvector.
    `VecP` : `State (constructor)`
        The constructor of the projection vector onto an eigenspace.
    
    Attributes
    ----------
    `q` : `State`
        The eigenvector of the geostrophic subspace.
    `p` : `State`
        The projection vector onto the geostrophic subspace.
    
    Methods
    -------
    `__call__(z: State) -> State`
        Project a state to the geostrophic subspace.
    """
    def __init__(self, mset: 'ModelSettingsBase',
                 VecQ, VecP) -> None:
        super().__init__(mset)
        # Construct the eigenvectors
        self.q = VecQ(0, mset)
        self.p = VecP(0, mset)
        return

    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project a state to the geostrophic subspace.
        
        Parameters
        ----------
        `z` : `State`
            The state to project.
        
        Returns
        -------
        `State`
            The projection of the state onto the geostrophic subspace.
        """
        return z.project(self.p, self.q)