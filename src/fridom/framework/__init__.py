"""Core modules of the FRIDOM framework.

Description
-----------
This module contain the base classes and functions for the FRIDOM framework.
This module should mainly be used for developing new modules and models.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # isort: off
    # Import classes
    from .model_settings_base import ModelSettingsBase
    from .field_variable import FieldVariable
    from .state_base import StateBase
    from .model_state import ModelState
    from .model import Model
    from .clock import Clock

    # Import logger
    from .logger import log
    # Import config
    from .configuration import config

    # Import modules
    from . import exceptions
    from . import grid
    from . import domain_decomposition
    from . import utils
    from . import time_steppers
    from . import modules
    from . import projection
    from . import timing_module

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom.framework": [
        "exceptions",
        "grid",
        "domain_decomposition",
        "utils",
        "time_steppers",
        "modules",
        "projection",
        "timing_module",
    ],
}

all_imports_by_origin = {
    "fridom.framework.configuration": ["config"],
    "fridom.framework.logger": ["log"],
    "fridom.framework.model_settings_base": ["ModelSettingsBase"],
    "fridom.framework.field_variable": ["FieldVariable"],
    "fridom.framework.state_base": ["StateBase"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
    "fridom.framework.clock": ["Clock"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
