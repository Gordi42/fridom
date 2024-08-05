# Import external modules
from typing import TYPE_CHECKING, Union
import numpy as np
# Import internal modules
import fridom.framework as fr
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
    _dynamic_attributes = set(["z", "z_diag", "dz", "it",
                               "_start_time", "_start_time_in_seconds",
                               "_passed_time"])
    def __init__(self, mset: 'ModelSettingsBase') -> None:
        self.mset = mset
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self.start_time = 0
        self._start_time_in_seconds = 0
        self._start_time = 0
        self._passed_time = 0
        self.time = 0
        # flag to cancel the model run in case something goes wrong
        self.panicked = False

    def reset(self) -> None:
        """
        Reset the model state.
        """
        self.z *= 0.0
        self.z_diag *= 0.0
        self.dz *= 0.0
        self.it = 0
        self.time = 0
        return

    # ================================================================
    #  xarray conversion
    # ================================================================
    @property
    def xr(self):
        """
        Model State as xarray dataset
        """
        return self.xrs[:]


    @property
    def xrs(self):
        """
        Model State of sliced domain as xarray dataset 
        """
        import xarray as xr
        def slicer(key):
            ds_z = self.z.xrs[key]
            ds_zd = self.z_diag.xrs[key]
            ds = xr.merge([ds_z, ds_zd])
            return ds
        return fr.utils.SliceableAttribute(slicer)

    # ================================================================
    #  Time handling
    # ================================================================
    def get_total_time(self, time) -> Union[np.datetime64, float]:
        if isinstance(self.start_time, np.datetime64):
            return self.start_time + np.timedelta64(int(time), 's')
        else:
            return self.time

    @property
    def start_time(self) -> Union[np.datetime64, float]:
        """
        Get the start time.
        """
        return self._start_time
    
    @start_time.setter
    def start_time(self, value: Union[np.datetime64, float]) -> None:
        """
        Set the start time.
        """
        self._start_time = value
        self._start_time_in_seconds = fr.utils.to_seconds(value)
        return

    @property
    def total_time(self) -> Union[np.datetime64, float]:
        """
        Return the total time either as a datetime object or as a float (seconds).
        """
        return self.get_total_time(self._passed_time)
    
    @property
    def time(self) -> float:
        """
        Get the model time.
        """
        return self._start_time_in_seconds + self._passed_time

    @time.setter
    def time(self, value: float) -> None:
        self._passed_time = value - self._start_time_in_seconds

utils.jaxify_class(ModelState)