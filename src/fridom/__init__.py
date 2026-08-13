"""
Framework for Idealized Ocean Models (FRIDOM).

==============================================

Description
-----------
FRIDOM is a modeling framework designed with a singular goal in mind:
to provide a high-level interface for the development of idealized
ocean models.
FRIDOM is built on JAX, enabling jit-compiled model execution on CPUs,
GPUs, and TPUs at medium resolutions, constrained only by your hardware
capabilities, right within Jupyter Notebook.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/FRIDOM
"""
from typing import TYPE_CHECKING

import jax
from lazypimp import setup

from fridom._compile_cache import configure as _configure_compile_cache

# ================================================================
#  JAX configuration
# ================================================================
# FRIDOM uses double precision by default. Users who prefer single
# precision can disable x64 after importing fridom (see the jax
# documentation on double precision). The compute platform (cpu/gpu/tpu)
# is selected through JAX directly, e.g. via the JAX_PLATFORMS
# environment variable.
#
# This belongs in the root package, and as early in it as possible:
# jax reads the flag when an array is created, so every array made
# before the flag is set is silently float32. Setting it here makes
# ``import fridom`` alone sufficient — no subpackage import needed —
# which is what keeps the default at float64 once the old
# ``fridom.framework`` tree (its previous home) is deleted.
jax.config.update("jax_enable_x64", val=True)

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from . import (
        benchmarking,
        framework,
        hydrostatic,
        io,
        model,
        nonhydro,
        ops,
        scaling,
        shallowwater,
        spatial,
    )

    # the time-law family, aliased at the root (the spelling the
    # docstrings across the package already use, e.g. ``fr.Ramp``)
    from .model.time_dependent import (
        Harmonic,
        Ramp,
        TimeDependent,
        TimeFunction,
        TimeSeries,
    )

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "fridom": [
        "benchmarking",
        "framework",
        "hydrostatic",
        "io",
        "model",
        "nonhydro",
        "ops",
        "scaling",
        "shallowwater",
        "spatial",
    ],
}

all_imports_by_origin = {
    "fridom.model.time_dependent": [
        "TimeDependent", "Ramp", "TimeFunction", "TimeSeries",
        "Harmonic"],
}

# Enable the persistent JAX compilation cache before any lazy import can
# trigger a compile (see fridom/_compile_cache.py); a no-op when already
# configured or disabled via FRIDOM_DISABLE_COMPILE_CACHE.
_configure_compile_cache()

setup(__name__, all_modules_by_origin, all_imports_by_origin)
