# Import external modules
from typing import TYPE_CHECKING, Union
import numpy as np
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
    _dynamic_attributes = set(["z", "z_diag", "dz", "it", "_time"])
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self._time = None # the time in seconds since the model start time
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

    @property
    def time(self):
        passed_time = np.timedelta64(int(self._time*1e9), 'ns')
        return self.z.mset.start_time + passed_time

    @time.setter
    def time(self, t: Union[np.datetime64, float]):
        if isinstance(t, np.datetime64):
            self._time = (t - self.z.mset.start_time).astype('float64')*1e-9
        else:
            self._time = t

utils.jaxify_class(ModelState)