from fridom.framework.grid_base import GridBase
from fridom.framework.modelsettings_base import ModelSettingsBase
from fridom.framework.state_base import StateBase
from fridom.framework.model_state import ModelStateBase
from fridom.framework.timing_module import TimingModule

from functools import wraps

def start_module(method):
    """
    Decorator for the start method of a module. 
    Sets the grid, mset, and timer as attributes of the module if the module is 
    enabled.
    """
    @wraps(method)
    def wrapper(self, **kwargs):
        if self.is_enabled():
            grid = kwargs.get('grid')
            self.grid = grid
            self.mset = grid.mset
            self.timer = kwargs.get('timer')
            method(self)
    return wrapper

def stop_module(method):
    """
    Decorator for the stop method of a module.

    """
    @wraps(method)
    def wrapper(self, **kwargs):
        if self.is_enabled():
            method(self)
    return wrapper

def update_module(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.is_enabled():
            self.timer.get(self.name).start()
            method(self, *args, **kwargs)
            self.timer.get(self.name).stop()
    return wrapper

class Module:
    """
    # Base class for all model modules.
    A module is a component of the model that is executed at each time step.
    It can for example be a tendency term, a parameterization, or a diagnostic
    as for example outputting the model state to a file.

    ## Required methods:
    1. `__init__(self, ...) -> None`: The constructor only takes keyword 
    argument which are stored as attributes. Always call the parent constructor 
    with `super().__init__(name, **kwargs)`. The name of the module is stored in
    the timing module and should not be too long.
    2. `update(self, mz: ModelStateBase, dz: StateBase) -> None`: This method is
    called by the model at each time step. It can for example update the 
    tendency state `dz` based on the model state `mz`. Or write the model state
    to a file. Make sure to wrap the method with the `@update_module` decorator.

    ## Optional methods:
    1. `start(self, grid: GridBase, timer: TimingModule) -> None`: This method is
    called by the model when the module is started. It can for example open an
    output file. Make sure to wrap the method with the `@start_module` decorator.
    2. `stop(self) -> None`: This method is called by the model when the module
    is stopped. It can for example close an output file. Make sure to wrap the
    method with the `@stop_module` decorator.

    ## Example:
    An example of a simple module that increments a number at each time step:
    ```python
    import fridom.framework as fr

    class Increment(fr.modules.Module):
        def __init__(self):
            # sets the module name to "Increment", and the number to None
            super().__init__("Increment", number=None)
    
        @fr.modules.start_module
        def start(self):
            self.number = 0  # sets the number to 0

        @fr.modules.update_module
        def update(self, mz: fr.ModelSettingsBase, dz: fr.StateBase) -> None:
            self.number += 1  # increments the number by 1

        @fr.modules.stop_module
        def stop(self):
            self.number = None  # sets the number to None
    ```
    """
    def __init__(self, name, **kwargs) -> None:
        """
        Initialize the module with the given keyword arguments.
        Child classes should always call the parent constructor with
        `super().__init__(name, **kwargs)`. Avoid giving the model settings or the grid
        as arguments, as they are set by the model when the module is started.
        """
        # The module is enabled by default
        self.__enabled = True
        # The grid should be set by the model when the module is started
        self.grid: GridBase = None
        self.mset: ModelSettingsBase = None
        self.timer: TimingModule = None
        # Update the attributes with the keyword arguments
        self.__dict__.update(kwargs)
        self.name = name
        return

    @start_module
    def start(self) -> None:
        """
        This method is called by the model when the module is started. Child 
        classes that require a start method (for example to start an output writer) 
        should overwrite this method. 

        ## Note:
        The start method should have no arguments. The grid and timer are set
        as attributes of the module when the start method is called. (See the
        `start_module` decorator.)
        """
        return

    @stop_module
    def stop(self) -> None:
        """
        This method is called by the model when the module is stopped. Child
        classes that require a stop method (for example to close an output file)
        should overwrite this method.

        ## Note:
        Child classes should have the same signature as this method, e.g.
        it can not have any arguments.
        """
        return

    @update_module
    def update(self, mz: ModelStateBase, dz: StateBase) -> None:
        """
        This method is called by the model at each time step. Child classes
        should overwrite this method to implement the module's functionality.

        ## Arguments:
        - `mz`: The model state at the current time step.
        - `dz`: The tendency state at the current time step.
        """
        pass

    def enable(self) -> None:
        """
        Enabling the module means that it will be executed at each time step.
        Disabled modules are neither initialized nor updated.
        """
        self.__enabled = True
        return
    
    def disable(self) -> None:
        """
        Enabling the module means that it will be executed at each time step.
        Disabled modules are neither initialized nor updated.
        """
        self.__enabled = False
        return

    def is_enabled(self) -> bool:
        """
        Return whether the module is enabled or not.
        """
        return self.__enabled

    def reset(self):
        """
        Reset the module to its initial state. This method is called by the model
        when the model is reset. 
        """
        self.stop()
        self.start(grid=self.grid, timer=self.timer)

    def __repr__(self) -> str:
        res = f"  {self.name}:"
        if not self.__enabled:
            res += " (disabled)"
        res += "\n"
        return res

