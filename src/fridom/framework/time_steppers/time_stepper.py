# Import external modules
from typing import TYPE_CHECKING, Union
from numpy import ndarray
import numpy as np
# Import internal modules
from fridom.framework.modules.module import Module
from fridom.framework import config
# Import type information
if TYPE_CHECKING:
    from fridom.framework.state_base import StateBase
    from fridom.framework.model_state import ModelState


class TimeStepper(Module):
    """
    Base class for all time steppers.
    
    Description
    -----------
    Required methods:
    1. `__init__(self, ...) -> None`: The constructor only takes keyword
    argument which are stored as attributes. Always call the parent constructor
    with `super().__init__(name, **kwargs)`.
    2. `update(self, mz: ModelStateBase) -> None`: This method is called by the
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
    
    Parameters
    ----------
    `name` : `str`
        The name of the time stepper.
    **kwargs
        Additional keyword arguments.
    
    """
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._dt = None

    def update(self, mz: 'ModelState'):
        raise NotImplementedError

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
        config.logger.warning(
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
            self._dt = config.dtype_real(value / np.timedelta64(1, 's'))
        return
