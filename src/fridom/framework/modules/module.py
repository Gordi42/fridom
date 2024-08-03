# Import external modules
from typing import TYPE_CHECKING
from functools import wraps
# Import internal modules
from fridom.framework import config
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.model_state import ModelState
    from fridom.framework.timing_module import TimingModule

def setup_module(method):
    """
    Decorator for the setup method of a module. 
    Sets mset as attribute of the module if the module is
    enabled.
    """
    @wraps(method)
    def wrapper(self, **kwargs):
        if self.is_enabled():
            # if the log level is set, change the log level for the module
            if self.log_level is not None:
                old_log_level = config.logger.level
                config.set_log_level(self.log_level.value)

            config.logger.verbose(f"Setup module: {self.name}")
            
            mset = kwargs.get('mset')
            self.mset = mset
            method(self)

            # if the log level was set, change it back to the old log level
            if self.log_level is not None:
                config.set_log_level(old_log_level)
    return wrapper

def module_method(method):
    """
    Decorator for the start, update and stop method of a module.

    Description
    -----------
    Sets the log level of the module if the log level is set and time 
    the duration of the method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.is_enabled():
            # if the log level is set, change the log level for the module
            if self.log_level is not None:
                old_log_level = config.logger.level
                config.set_log_level(self.log_level.value)

            config.logger.debug(
                f"Calling '{method.__name__}' of: {self.name}")

            with self.mset.timer[self.name]:
                result = method(self, *args, **kwargs)

            # if the log level was set, change it back to the old log level
            if self.log_level is not None:
                config.set_log_level(old_log_level)
            return result
        else:
            if method.__name__ == "update":
                return kwargs.get('mz')
    return wrapper

class Module:
    """
    Base class for all modules.
    
    Description
    -----------
    A module is a component of the model that is executed at each time step.
    It can for example be a tendency term, a parameterization, or a diagnostic
    as for example outputting the model state to a file.

    Required methods:
    1. `__init__(self, ...) -> None`: The constructor only takes keyword 
    argument which are stored as attributes. Always call the parent constructor 
    with `super().__init__(name, **kwargs)`. The name of the module is stored in
    the timing module and should not be too long.
    2. `update(self, mz: ModelState) -> None`: This method is
    called by the model at each time step. It can for example update the 
    tendency state `mz.dz` based on the model state `mz`. Or write the model state
    to a file. Make sure to wrap the method with the `@update_module` decorator.

    Optional methods:
    1. `start(self, mset: ModelSettingsBase) -> None`: 
    This method is called by the model when the module is started. It can for 
    example open an output file. Make sure to wrap the method with the 
    `@start_module` decorator.
    2. `stop(self) -> None`: This method is called by the model when the module
    is stopped. It can for example close an output file. Make sure to wrap the
    method with the `@stop_module` decorator.
    
    Parameters
    ----------
    `name` : `str`
        The name of the module.
    `**kwargs`
        Keyword arguments that are stored as attributes of the module.
    
    Flags
    -----
    `required_halo` : `int`
        The number of halo points required by the module.
    `mpi_available` : `bool`
        Whether the module can be run in parallel.
    `execute_at_start` : `bool`
        Whether the module should be executed before the first time step.
    
    Examples
    --------
    >>> :# Provide examples of how to use this class.}
    >>> import fridom.framework as fr
    >>> class Increment(fr.modules.Module):
    ...    def __init__(self):
    ...        # sets the module name to "Increment", and the number to None
    ...        super().__init__("Increment", number=None)
    ...    @fr.modules.start_module
    ...    def start(self):
    ...        self.number = 0  # sets the number to 0
    ...    @fr.modules.update_module
    ...    def update(self, mz: fr.ModelSettingsBase) -> None:
    ...        self.number += 1  # increments the number by 1
    ...    @fr.modules.stop_module
    ...    def stop(self):
    ...        self.number = None  # sets the number to None
    """
    def __init__(self, name, **kwargs) -> None:
        # The module is enabled by default
        self.__enabled = True
        # The log level
        self.log_level: config.LogLevel | None = None
        # Set the flags
        self.required_halo = 0  # The required halo for the module
        self.mpi_available = True  # Whether the module can be run in parallel
        self.execute_at_start = False
        # The grid should be set by the model when the module is started
        self.mset: 'ModelSettingsBase | None' = None
        self.timer: 'TimingModule | None' = None
        # Update the attributes with the keyword arguments
        self.__dict__.update(kwargs)
        self.name = name
        return

    @setup_module
    def setup(self) -> None:
        """
        Start the module

        Description
        -----------
        This method is called by the ModelSettings.setup() and sets the 
        ModelSettings as an attribute. Make sure to decorate the method with
        the `@setup_module` decorator

        .. note::
        ----
        The setup method should have no arguments. The model settings are set
        as attributes of the module when the setup method is called. (See the
        `setup_module` decorator.)
        """
        return

    def start(self) -> None:
        """
        Start the module

        Description
        -----------
        This method is called at the beginning of the model run. Child classes 
        that require a start method (for example to start an output writer)
        should overwrite this method. Make sure to decorate the method with
        the `@module_method` decorator.
        """
        return

    def stop(self) -> None:
        """
        Stop the module

        Description
        -----------
        This method is called by the model at the end of the model run or
        when the model is reset. Child classes that require a stop method 
        (for example to close an output file) should overwrite this method.
        Make sure to decorate the method with the `@module_method` decorator.
        """
        return

    def update(self, mz: 'ModelState') -> 'ModelState':
        """
        Update the module
        
        Description
        -----------
        This method is called by the model at each time step. Child classes
        should overwrite this method to update the module. Make sure to decorate
        the method with the `@module_method` decorator.
        
        Parameters
        ----------
        `mz` : `ModelState`
            The model state at the current time step.

        Returns
        -------
        `ModelState`
            The updated model state.
        """
        return mz
        

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
        Stop and start the module.
        """
        self.stop()
        self.start()

    def __repr__(self) -> str:
        """
        String representation of the time stepper.
        """
        res = self.name
        if not self.__enabled:
            res += " (disabled)"

        for key, value in self.info.items():
            res += "\n  - {}: {}".format(key, value)
        return res

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def info(self) -> dict:
        """
        Return a dictionary with information about the time stepper.
        
        Description
        -----------
        This method should be overridden by the child class to return a
        dictionary with information about the time stepper. This information is
        used to print the time stepper in the `__repr__` method.
        """
        return {}

    @property
    def grid(self) -> 'GridBase':
        """
        Return the grid of the model.
        """
        return self.mset.grid
