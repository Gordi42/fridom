import fridom.framework as fr
from functools import partial
# Import external modules
from typing import TYPE_CHECKING, Union
import numpy as np
# Import internal modules
from fridom.framework import utils
from fridom.framework.config import logger
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase


@partial(utils.jaxify, dynamic=('grid',))
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

    And the following methods:
    - state_constructor
    - diagnostic_state_constructor
    
    Examples
    --------
    Create a new model settings class by inheriting from `ModelSettingsBase`:

    .. code-block:: python

        import fridom.framework as fr
        class ModelSettings(fr.ModelSettingsBase):
            def __init__(self, grid, **kwargs):
                super().__init__(grid)
                self.model_name = "MyModel"
                # set other parameters
                self.my_parameter = 1.0
                # Finally, set attributes from keyword arguments
                self.set_attributes(**kwargs)

            # optional: override the parameters property
            @property
            def parameters(self):
                res = super().parameters
                res["my_parameter"] = self.my_parameter
                return res
    """
    model_name      = "Unnamed model"

    def __init__(self, grid: 'GridBase', **kwargs) -> None:
        self._time_stepper   = fr.time_steppers.AdamBashforth()
        self._progress_bar   = fr.modules.ProgressBar()
        self._tendencies     = fr.modules.ModuleContainer("All Tendencies")
        self._diagnostics    = fr.modules.ModuleContainer("All Diagnostics")
        self._restart_module = fr.modules.RestartModule()
        self._timer          = fr.timing_module.TimingModule()
        self._nan_check_interval = 100
        self._custom_fields  = []
        self._halo           = None
        self.grid = grid
        self.set_attributes(**kwargs)
        return

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

    def setup_grid(self):
        """
        Setup the grid object.
        """
        self.grid.setup(mset=self)
        return
    
    def setup_all_modules(self):
        self.grid.water_mask.setup(mset=self)
        self.progress_bar.setup(mset=self)
        self.restart_module.setup(mset=self)
        self.tendencies.setup(mset=self)
        self.diagnostics.setup(mset=self)
        self.time_stepper.setup(mset=self)
        return

    def setup_settings_parameters(self):
        """
        Setup the model settings parameters.
        """
        return

    def setup(self):
        logger.verbose("Setting up model settings")
        self.setup_grid()
        self.setup_settings_parameters()
        self.setup_all_modules()
        logger.info(self)
        return

    def state_constructor(self):
        from fridom.framework.state_base import StateBase
        return StateBase(self, {})

    def diagnostic_state_constructor(self):
        from fridom.framework.state_base import StateBase
        return StateBase(self, {})

    def __repr__(self) -> str:
        """
        String representation of the model settings (for IPython).
        """
        return f"""
=================================================
  Model Settings:
-------------------------------------------------
# {self.model_name}
# Parameters: {self.__parameters_to_string()}
# Grid: {self.grid}
# Time Stepper: {self.time_stepper}
# {self.restart_module}
# Tendencies: {self.tendencies}
# Diagnostics: {self.diagnostics}
=================================================
        """

    def add_field_to_state(self, kwargs: dict) -> None:
        """
        Add a field variable to the state vector.

        Description
        -----------
        This method can be used to extend the state vector with a new field 
        variable, for example when adding a new tracer to the model.

        Parameters
        ----------
        `kwargs` : `dict`
            Dictionary that contains the arguments required to construct 
            the field.
        """
        # check if a name is provided
        if "name" not in kwargs: 
            logger.critical("Error occurred while adding a field to the state.")
            logger.critical("Field name not provided")
            logger.critical("Please provide a name in the kwargs dictionary.")
            raise ValueError
        name = kwargs["name"]
        all_names = [field["name"] for field in self.custom_fields]
        # check if the field name already exists
        if name in all_names:
            logger.critical("Error occurred while adding a field to the state.")
            logger.critical(f"Field name {name} already exists")
            logger.critical(f"Used names: {all_names}")
            logger.critical("Please provide a unique name in the kwargs dictionary.")
            raise ValueError

        self.custom_fields.append(kwargs)
        return

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:
        """
        Return a dictionary with all parameters of the model settings.

        Description
        -----------
        This method should be overridden by the child class to return a dictionary
        with all parameters of the model settings. This dictionary is used to print
        the model settings in the `__repr__` method.
        """
        return {}

    def __parameters_to_string(self):
        # res = "\n"
        res = ""
        for key, value in self.parameters.items():
            res += "\n  - {}: {}".format(key, value)
        return res

    @property
    def grid(self) -> 'GridBase':
        """
        The spatial grid.
        """
        return self._grid

    @grid.setter
    def grid(self, value: 'GridBase') -> None:
        self._grid = value
        return

    # ----------------------------------------------------------------
    #  Module properties
    # ----------------------------------------------------------------

    @property
    def time_stepper(self):
        """
        The time stepper object (default: AdamBashforth).
        """
        return self._time_stepper

    @time_stepper.setter
    def time_stepper(self, value):
        self._time_stepper = value
        return

    @property
    def progress_bar(self):
        """
        The progress bar object (default: ProgressBar).
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, value):
        self._progress_bar = value
        return

    @property
    def tendencies(self):
        """
        The module container for all tendencies.
        """
        return self._tendencies
    
    @tendencies.setter
    def tendencies(self, value):
        self._tendencies = value
        return
    
    @property
    def diagnostics(self):
        """
        The module container for all diagnostics.
        """
        return self._diagnostics
    
    @diagnostics.setter
    def diagnostics(self, value):
        self._diagnostics = value
        return
    
    @property
    def restart_module(self):
        """
        The restart module.
        """
        return self._restart_module
    
    @restart_module.setter
    def restart_module(self, value):
        self._restart_module = value
        return

    @property
    def timer(self):
        """
        The timing module.
        """
        return self._timer
    
    @timer.setter
    def timer(self, value):
        self._timer = value
        return

    # ----------------------------------------------------------------
    #  Other properties
    # ----------------------------------------------------------------

    @property
    def nan_check_interval(self) -> int:
        """
        The interval at which the model checks for NaN values
        """
        return self._nan_check_interval

    @nan_check_interval.setter
    def nan_check_interval(self, value: int) -> None:
        self._nan_check_interval = value
        return

    @property
    def custom_fields(self) -> list:
        """
        List of custom fields to be added to the state vector.
        """
        return self._custom_fields
    
    @custom_fields.setter
    def custom_fields(self, value: list) -> None:
        self._custom_fields = value
        return


    @property
    def halo(self) -> int:
        """
        Return the halo size of the model.
        """
        if self._halo is not None:
            return self._halo
        return self.tendencies.required_halo

    @halo.setter
    def halo(self, value: int) -> None:
        self._halo = value
        return