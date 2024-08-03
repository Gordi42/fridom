# Import external modules
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase

class Projection:
    """
    Base class for projections. All projections should inherit from this class.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    """
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.mset = mset
        self.grid = mset.grid

    @abstractmethod
    def __call__(self, z: 'StateBase') -> 'StateBase':
        """
        Project the state on the given subspace.

        Parameters
        ----------
        `z` : `State`
            The state to project.

        Returns
        -------
        `State`
            The projected state.
        """
        pass