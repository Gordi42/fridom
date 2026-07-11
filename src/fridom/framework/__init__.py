"""
Core modules of the FRIDOM framework.

Description
-----------
This module contain the base classes and functions for the FRIDOM framework.
This module should mainly be used for developing new modules and models.
"""
from typing import TYPE_CHECKING

import jax
from lazypimp import setup

# ================================================================
#  JAX configuration
# ================================================================
# FRIDOM uses double precision by default. Users who prefer single
# precision can disable x64 after importing fridom (see the jax
# documentation on double precision). The compute platform (cpu/gpu/tpu)
# is selected through JAX directly, e.g. via the JAX_PLATFORMS
# environment variable.
jax.config.update("jax_enable_x64", val=True)

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
    from .field_base import FieldBase
    from .field_metadata import FieldMetadata
    from .logger import log
    from .model import Model
    from .model_settings_base import ModelSettingsBase
    from .model_state import ModelState
    from .scalar_field import ScalarField
    from .tensor_field import TensorField
    from .vector_field import VectorField

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
    "fridom.framework.logger": ["log"],
    "fridom.framework.model_settings_base": ["ModelSettingsBase"],
    "fridom.framework.field_base": ["FieldBase"],
    "fridom.framework.field_metadata": ["FieldMetadata"],
    "fridom.framework.scalar_field": ["ScalarField"],
    "fridom.framework.vector_field": ["VectorField"],
    "fridom.framework.tensor_field": ["TensorField"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
    "fridom.framework.clock": ["Clock", "TimingFormat"],
    "fridom.framework.clock_trigger": ["ClockTrigger"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)

# ================================================================
#  Job initialization
# ================================================================
from fridom.framework.logger import log as _log  # noqa: E402

# On multi-host setups, only the main process should log.
if jax.process_count() > 1:
    if jax.process_index() == 0:
        _log.setLevel("INFO")
    else:
        _log.setLevel("SILENT")

from fridom.framework.utils.printing import print_job_init_info  # noqa: E402

print_job_init_info()
