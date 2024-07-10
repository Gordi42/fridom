# Import external modules
from typing import TYPE_CHECKING
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase
    from fridom.framework.time_steppers.time_stepper import TimeStepper

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
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self.time = mset.start_time