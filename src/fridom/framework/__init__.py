"""
# FRIDOM Framework
This module contain the base classes and functions for the FRIDOM framework.
This module should mainly be used for developing new modules and models.

## Classes:
    - ModelSettingsBase: Base class for model settings
    - GridBase: Base class for grids
    - FieldVariable: Class for storing scalar variables
    - StateBase: Base class for storing state variables
    - ModelStateBase: Base class for storing model state variables
    - ModelBase: Base class for models
    - NetCDFWriter: Class for writing netcdf files
    - DiagnoseImbalanceBase: Base class for diagnosing imbalances

## Modules:
    - time_steppers: Contains the base class for time steppers
    - modules: Contains the base class for modules
    - projection: Contains functions for flow decomposition
    - timing_module: Contains functions for timing the model (benchmarking)
"""

# import classes
from .modelsettings import ModelSettingsBase
from .grid_base import GridBase
from .field_variable import FieldVariable
from .state_base import StateBase
from .model_state import ModelState
from .model import Model
from .diagnose_imbalance_base import DiagnoseImbalanceBase

# import modules:
from . import time_steppers
from . import modules
from . import projection
from . import timing_module
from . import domain_decomposition