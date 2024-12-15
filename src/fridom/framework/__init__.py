"""
Core modules of the FRIDOM framework.

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
if TYPE_CHECKING:  # pragma: no cover
    # Import modules
    from . import (
        domain_decomposition,
        exceptions,
        grid,
        modules,
        projection,
        time_steppers,
        timing_module,
        utils,
    )

    # Import classes and functions
    from .clock import Clock, TimingFormat
    from .clock_trigger import ClockTrigger
    from .configuration import config
    from .field_metadata import FieldMetadata
    from .field_variable import FieldVariable
    from .logger import log
    from .model import Model
    from .model_settings_base import ModelSettingsBase
    from .model_state import ModelState
    from .state_base import StateBase

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
    "fridom.framework.field_metadata": ["FieldMetadata"],
    "fridom.framework.field_variable": ["FieldVariable"],
    "fridom.framework.state_base": ["StateBase"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
    "fridom.framework.clock": ["Clock", "TimingFormat"],
    "fridom.framework.clock_trigger": ["ClockTrigger"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
