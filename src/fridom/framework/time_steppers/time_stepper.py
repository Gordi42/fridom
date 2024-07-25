# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules.module import Module
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

    def update(self, mz: 'ModelState'):
        raise NotImplementedError

