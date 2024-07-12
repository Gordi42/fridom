"""
FRIDOM Framework
================

Description
-----------
This module contain the base classes and functions for the FRIDOM framework.
This module should mainly be used for developing new modules and models.

Modules
-------
'grid'
    Contains classes for grid generation
'domain_decomposition'
    Contains classes for domain decomposition (parallelization)
'config'
    Contains the configuration settings for the framework
'utils'
    Contains utility functions
'time_steppers'
    Contains classes related to time stepping
'modules'
    Contains the base class for modules
'projection'
    Contains functions for flow decomposition
'timing_module'
    Contains functions for timing the model (benchmarking)

Classes
-------
'FridomLogger'
    Class for logging messages
'LogLevel
    Enum class for the different log levels
'ModelSettingsBase'
    Base class for model settings
'FieldVariable'
    Class for storing scalar variables
'StateBase'
    Base class for storing state variables
'ModelState'
    Class for storing model state variables
'Model'
    Main class for models
'DiagnoseImbalanceBase'
    Base class for diagnosing imbalances
'MeshPoint'
    Enum class for the different types of mesh points

Functions
'to_numpy(x)'
    returns a deepcopy of x where all cupy arrays are converted to numpy arrays

"""

# import classes
from .model_settings_base import ModelSettingsBase
from .field_variable import FieldVariable
from .state_base import StateBase
from .model_state import ModelState
from .model import Model
from .diagnose_imbalance_base import DiagnoseImbalanceBase
from .mesh_point import MeshPoint

# import modules
from . import grid
from . import domain_decomposition
from . import config
from . import utils
from . import time_steppers
from . import modules
from . import projection
from . import timing_module

# import functions
from .to_numpy import to_numpy