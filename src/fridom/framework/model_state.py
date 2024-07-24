# Import external modules
from typing import TYPE_CHECKING
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
    _dynamic_attributes = set(["z", "z_diag", "dz", "it", "_time_array"])
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self._time_array = None
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
        tt = self._time_array
        # to string
        ds = f"{tt[0]:04d}-{tt[1]:02d}-{tt[2]:02d}T{tt[3]:02d}:{tt[4]:02d}:{tt[5]:02d}"
        if len(tt) > 6:
            ds += "."
            for i in range(6, len(tt), 1):
                ds += f"{tt[i]:03d}"
        return np.datetime64(ds)

    @time.setter
    def time(self, t: np.datetime64):
        date_str = str(t)
        date_part, time_part = date_str.split('T')
        part1 = date_part.split('-')
        hour, minute, second = time_part.split(':')
        if '.' in second:
            second, rest = second.split('.')
            rest = [int(rest[i:i+3]) for i in range(0, len(rest), 3)]
        else:
            rest = []
        self._time_array = np.array(list(part1) + [hour, minute, second] + rest, 
                                    dtype=np.int64)

utils.jaxify_class(ModelState)