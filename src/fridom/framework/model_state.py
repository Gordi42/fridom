# Import external modules
from typing import TYPE_CHECKING
# Import type information
if TYPE_CHECKING:
    from fridom.framework.modelsettings_base import ModelSettingsBase
    from fridom.framework.state_base import StateBase

class ModelState:
    """
    The base class for model states. It contains the state vector, the time step
    and the model time. Child classes may add more attributes as for example the
    diagnostic variables needed for the model.

    All model state variables should be stored in this class.

    ## Attributes
    - z (StateBase)   : State vector
    - it (int)        : Time step
    - time (float)    : Model time
    """
    def __init__(self, 
                 mset: 'ModelSettingsBase', 
                 initialize_state = True) -> None:
        """
        The base constructor for the ModelStateBase class.
        """
        if initialize_state:
            self.z = mset.state_constructor()
            self.z_diag = mset.diagnostic_state_constructor()
        else:
            self.z: StateBase = None
            self.z_diag: StateBase = None
        self.it = 0
        self.time = 0.0