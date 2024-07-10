# Import external modules
from typing import TYPE_CHECKING
import numpy as np
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase

class ModelSettingsBase:
    """
    Base class for model settings container.
    
    Description
    -----------
    This class should be used as a base class for all model settings containers.
    It provides a set of attributes and methods that are common to all models.
    Child classes should override the following attributes: 
    - n_dims 
    - model_name
    - tendencies
    - diagnostics
    - state_constructor
    - diagnostic_state_constructor

    
    Attributes
    ----------
    `model_name` : `str` (default: `"Unnamed model"`)
        The name of the model.
    `time_stepper` : `TimeStepperBase` (default: `AdamBashforth()`
        The time stepper object.
    `tendencies` : `ModuleContainer`
        A container for all modules that calculate tendencies.
    `diagnostics` : `ModuleContainer`
        A container for all modules that calculate diagnostics.
    `bc` : `BoundaryConditions`
        The boundary conditions object.
    `state_constructor` : `callable`
        A function that constructs a state from the model settings
    `diagnostic_state_constructor` : `callable`
        A function that constructs a diagnostic state from the model settings
    `enable_verbose` : `bool` (default: `False`)
        Enable verbose output.
    
    
    Methods
    -------
    `print_verbose(*args, **kwargs)`
        Print function for verbose output.
    `set_attributes(**kwargs)`
        Set attributes from keyword arguments.
    
    Examples
    --------
    Create a new model settings class by inheriting from `ModelSettingsBase`:

    >>> from fridom.framework import ModelSettingsBase
    >>> class ModelSettings(ModelSettingsBase):
    ...     def __init__(self, grid, **kwargs):
    ...         super().__init__(grid)
    ...         self.model_name = "MyModel"
    ...         # set other parameters
    ...         self.my_parameter = 1.0
    ...         # set up modules and state constructors here. Eee for example 
    ...         # the `ModelSettings` class in `shallowwater/model_settings.py`
    ...         # Finally, set attributes from keyword arguments
    ...         self.set_attributes(**kwargs)
    ...     # maybe update the __str__ method to include the new parameter
    ...     def __str__(self) -> str:
    ...         res = super().__str__()
    ...         res += "  My parameter: {}\\n".format(self.my_parameter)
    ...         return res
    >>> settings = ModelSettings(L=[4*np.pi, 4*np.pi], my_parameter=2.0)
    >>> print(settings)
    """
    def __init__(self, grid: 'GridBase', **kwargs) -> None:
        # grid
        self.grid = grid

        # model name
        self.model_name = "Unnamed model"

        # time parameters
        from fridom.framework.time_steppers.adam_bashforth import AdamBashforth
        self.time_stepper = AdamBashforth()

        # modules
        from fridom.framework.modules.module_container import ModuleContainer
        from fridom.framework.modules.boundary_conditions import BoundaryConditions
        # List of modules that calculate tendencies
        self.tendencies = ModuleContainer(name="All Tendency Modules")
        # List of modules that do diagnostics
        self.diagnostics = ModuleContainer(name="All Diagnostic Modules")
        # Boundary conditions  (should be set by the child class)
        self.bc = BoundaryConditions(field_names=[])

        # Output
        self.enable_verbose = False   # Enable verbose output
        
        # Starttime
        self.start_time = np.datetime64(0, 's')

        # State Constructors
        from fridom.framework.state_base import StateBase
        self.state_constructor = lambda: StateBase(self, [])
        self.diagnostic_state_constructor = lambda: StateBase(self, [])

        # Set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def set_attributes(self, **kwargs):
        """
        Set model settings attributes from keyword arguments. If an attribute
        does not exist, an AttributeError is raised.
        
        Parameters
        ----------
        `**kwargs` : `dict`
            Keyword arguments to set the attributes of the model settings.
        
        Raises
        ------
        `AttributeError`
            The attribute does not exist in the model settings.
        """
        # Set attributes from keyword arguments
        for key, value in kwargs.items():
            # Check if attribute exists
            if not hasattr(self, key):
                raise AttributeError(
                    "ModelSettings has no attribute '{}'".format(key)
                    )
            setattr(self, key, value)
        return

    def setup(self):
        self.grid.setup(self)
        return

    def print_verbose(self, *args, **kwargs):
        """
        Print only if verbose output is enabled.
        """
        if self.enable_verbose:
            print(*args, **kwargs)
        return

    def __str__(self) -> str:
        """
        String representation of the model settings.
        """
        res = "================================================\n"
        res += "  Model Settings:\n"
        res += "================================================\n"
        res += "  Model name: {}\n".format(self.model_name)
        res += f"{self.time_stepper}"
        res += f"{self.tendencies}"
        res += f"{self.diagnostics}"
        res += "------------------------------------------------\n"
        return res

    def __repr__(self) -> str:
        """
        String representation of the model settings (for IPython).
        """
        return self.__str__()
