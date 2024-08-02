# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.projection.projection import Projection
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
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
    `use_discrete` : `bool` (default: `True`)
        Whether to use discrete eigenvectors.
    """
    def __init__(self, 
                 mset: 'ModelSettingsBase',
                 VecQ: 'StateBase',
                 VecP: 'StateBase',
                 use_discrete: bool = True) -> None:
        super().__init__(mset)
        self.q = VecQ(mset, s="d", use_discrete=use_discrete)
        self.p = VecP(mset, s="d", use_discrete=use_discrete)
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