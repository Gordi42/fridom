import fridom.framework as fr
from typing import Union
from numpy import ndarray
import numpy as np


class TimeStepper(fr.modules.Module):
    """
    Base class for all time steppers.
    
    Description
    -----------
    Required methods:
    1. `__init__(self, ...) -> None`: The constructor only takes keyword
    argument which are stored as attributes. Always call the parent constructor
    with `super().__init__(name, **kwargs)`.
    2. `update(self, mz: ModelState) -> None`: This method is called by the
    model at each time step. It should update the model state `mz` to the next
    time level. Make sure to wrap the method with the `@update_module` decorator
    from the `Module` class.

    Optional methods:
    1. `start(self) -> None`: This method is called by the model before the
    time stepping starts. It can be used to initialize variables or allocate
    memory. Make sure to wrap the method with the `@start_module` decorator from
    the `Module` class.
    2. `stop(self) -> None`: This method is called by the model after the time
    stepping has finished. It can be used to deallocate memory or clean up.
    Make sure to wrap the method with the `@stop_module` decorator from the
    `Module` class.
    """
    name = "Time Stepper"
    def __init__(self) -> None:
        super().__init__()
        self._dt = None  # set the time step size
        return

    def time_discretization_effect(self, omega: ndarray) -> ndarray:
        """
        Compute the time discretization effect on a frequency.
        
        Parameters
        ----------
        `omega` : `ndarray | float | complex`
            The frequency of the wave.
        
        Returns
        -------
        `ndarray`
            The frequency of the wave including the time discretization effect.
        """
        fr.log.warning(
            f"The time stepper {self.name} has no method to compute the time discretization effect."
        )
        return omega

    @property
    def info(self) -> dict:
        res = super().info
        res["dt"] = f"{self.dt} s"
        return res

    @property
    def dt(self) -> np.timedelta64:
        """
        Time step size.
        """
        return self._dt

    @dt.setter
    def dt(self, value: Union[np.timedelta64, float]) -> None:
        if isinstance(value, float) or isinstance(value, int):
            self._dt = value
        else:
            self._dt = fr.config.dtype_real(value / np.timedelta64(1, 's'))
        return
