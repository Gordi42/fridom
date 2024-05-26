"""
# Model Modules for the Nonhydrostatic Model
See fridom/framework/modules/__init__.py for more information.

## Available Modules:
    - LinearTendency: Linear tendency term
    - PressureGradientTendency: Pressure gradient tendency term
    - PressureSolve: Pressure solver
    - SourceTendency: Source term tendency
    - NetCDFWriter: NetCDF writer module
    - Diagnostics: Diagnostic module
    - animation: Contains animation modules
    - advection: Contains advection modules
    - interpolation: Contains interpolation modules
    - diffusion: Contains diffusion modules
"""

from .linear_tendency import LinearTendency
from .pressure_gradient_tendency import PressureGradientTendency
from .pressure_solve import PressureSolve
from .source_tendency import SourceTendency
from .netcdf_writer import NetCDFWriter
from .diagnostics import Diagnostics

# move the animation module into the nonhydro namespace
from fridom.framework.modules import animation

from . import advection
from . import interpolation
from . import diffusion