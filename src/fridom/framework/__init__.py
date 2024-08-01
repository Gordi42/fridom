"""
Core modules of the FRIDOM framework.

Description
-----------
This module contain the base classes and functions for the FRIDOM framework.
This module should mainly be used for developing new modules and models.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import classes
    from .model_settings_base import ModelSettingsBase
    from .field_variable import FieldVariable
    from .state_base import StateBase
    from .model_state import ModelState
    from .model import Model
    from .diagnose_imbalance_base import DiagnoseImbalanceBase

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

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom.framework": [
        "grid",
        "domain_decomposition",
        "config",
        "utils",
        "time_steppers",
        "modules",
        "projection",
        "timing_module",
    ],
}

all_imports_by_origin = {
    "fridom.framework.model_settings_base": ["ModelSettingsBase"],
    "fridom.framework.field_variable": ["FieldVariable"],
    "fridom.framework.state_base": ["StateBase"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
    "fridom.framework.diagnose_imbalance_base": ["DiagnoseImbalanceBase"],
    "fridom.framework.to_numpy": ["to_numpy"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)