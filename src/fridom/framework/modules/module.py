import fridom.framework as fr
from functools import wraps

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
                old_log_level = fr.log.level
                fr.config.set_log_level(self.log_level.value)

            fr.log.debug(
                f"Calling '{method.__name__}' of: {self.name}")

            # check if the model settings are already set
            if self.mset is None:
                result = method(self, *args, **kwargs)
            else:
                with self.mset.timer[self.name]:
                    result = method(self, *args, **kwargs)

            # if the log level was set, change it back to the old log level
            if self.log_level is not None:
                fr.config.set_log_level(old_log_level)
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

    .. code-block:: python

        import fridom.framework as fr
        class Increment(fr.modules.Module):
           def __init__(self):
               # sets the module name to "Increment", and the number to None
               super().__init__("Increment", number=None)
           @fr.modules.start_module
           def start(self):
               self.number = 0  # sets the number to 0
           @fr.modules.update_module
           def update(self, mz: fr.ModelSettingsBase) -> None:
               self.number += 1  # increments the number by 1
           @fr.modules.stop_module
           def stop(self):
               self.number = None  # sets the number to None
    """
    name = "Base Module"
    _is_mod_submodule = False
    def __init__(self) -> None:
        # The module is enabled by default
        self.__enabled = True
        # The log level
        self.log_level: fr.config.LogLevel | None = None
        # Set the flags
        self._required_halo = None  # The number of halo points required by the module
        self.mpi_available = True  # Whether the module can be run in parallel
        self.execute_at_start = False
        # The grid should be set by the model when the module is started
        self.mset: 'fr.ModelSettingsBase | None' = None
        self.timer: 'fr.timing_module.TimingModule | None' = None
        # Differentiation and interpolation modules
        self._diff_module: 'fr.grid.DiffModule | None' = None
        self._interp_module: 'fr.grid.InterpolationModule | None' = None
        return

    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        """
        Start the module

        Description
        -----------
        This method is called by the ModelSettings.setup() and sets the 
        ModelSettings as well as the differentiation and interpolation modules. 
        """
        self.mset = mset
        if not self._is_mod_submodule:
            # setup the differentiation and interpolation modules
            self._setup_submodule("diff_module", mset)
            self._setup_submodule("interp_module", mset)
        return

    def _setup_submodule(self, name, mset):
        submodule = getattr(self, name)
        if submodule is None:
            submodule = getattr(mset.grid, name)
            setattr(self, name, submodule)
        else:
            submodule.setup(mset=mset)
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

    def update(self, mz: 'fr.ModelState') -> 'fr.ModelState':
        """
        Update the module
        
        Description
        -----------
        This method is called by the model at each time step. Child classes
        should overwrite this method to update the module. Make sure to decorate
        the method with the `@module_method` decorator.
        
        Parameters
        ----------
        `mz` : `fr.ModelState`
            The model state at the current time step.

        Returns
        -------
        `fr.ModelState`
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
        info = {}
        # Check if the grid is set
        grid_is_set = self.mset is not None

        # ----------------------------------------------------------------
        #  Check if the differentiation module should be printed
        # ----------------------------------------------------------------

        print_diff = False
        diff_is_set = self.diff_module is not None
        if diff_is_set and grid_is_set:
            if self.diff_module is not self.grid.diff_module:
                print_diff = True
        elif diff_is_set:
            print_diff = True
        if print_diff:
            info["Diff. Module"] = self.diff_module.name

        # ----------------------------------------------------------------
        #  Check if the differentiation module should be printed
        # ----------------------------------------------------------------

        print_interp = False
        interp_is_set = self.interp_module is not None
        if interp_is_set and grid_is_set:
            if self.interp_module is not self.grid.interp_module:
                print_interp = True
        elif interp_is_set:
            print_interp = True
        if print_interp:
            info["Interp. Module"] = self.interp_module.name

        # ----------------------------------------------------------------
        #  Check if the required halo should be printed
        # ----------------------------------------------------------------

        print_halo = self._required_halo is not None
        if print_halo:
            info["Required Halo"] = self.required_halo
        return info

    @property
    def mset(self) -> 'fr.ModelSettingsBase':
        """
        The model settings
        """
        return self._mset
    
    @mset.setter
    def mset(self, mset: 'fr.ModelSettingsBase') -> None:
        self._mset = mset
        return

    @property
    def grid(self) -> 'fr.grid.GridBase':
        """
        The grid of the model settings
        """
        return self.mset.grid

    @property
    def diff_module(self) -> 'fr.grid.DiffModule':
        """The differentiation module to be used by this module."""
        return self._diff_module
    
    @diff_module.setter
    def diff_module(self, value):
        self._diff_module = value
        return

    @property
    def interp_module(self) -> 'fr.grid.InterpolationModule':
        """The interpolation module to be used by this module."""
        return self._interp_module

    @interp_module.setter
    def interp_module(self, value):
        self._interp_module = value
        return

    @property
    def required_halo(self) -> int:
        # Return the required halo if it is set
        if self._required_halo is not None:
            return self._required_halo
        # If it is not set, check the differentiation and interpolation modules
        # If they are not set, return 0
        req_halo = 0
        if self.diff_module is not None:
            req_halo = self.diff_module.required_halo
        if self.interp_module is not None:
            req_halo = max(req_halo, self.interp_module.required_halo)
        return req_halo
    
    @required_halo.setter
    def required_halo(self, value: int):
        self._required_halo = value
        return