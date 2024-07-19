# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import utils
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase

class ModelState:
    """
    Stores the model state variables and the time information.
    
    Description
    -----------
    The base class for model states. It contains the state vector, the time step
    and the model time. Child classes may add more attributes as for example the
    diagnostic variables needed for the model. All model state variables should be stored in this class.
    
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings object.
    
    Attributes
    ----------
    `z` : `State`
        The state vector with the state variables.
    `z_diag` : `State`
        The state vector with the diagnostic variables.
    `dz` : `State`
        The state vector tendency.
    """
    _dynamic_attributes = ["z", "z_diag", "dz", "it"]
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self.time = mset.start_time

    def reset(self) -> None:
        """
        Reset the model state.
        """
        self.z *= 0.0
        self.z_diag *= 0.0
        self.dz *= 0.0
        self.it = 0
        self.time = self.z.mset.start_time
        return

utils.jaxify_class(ModelState)
