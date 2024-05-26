"""
# Model Modules for the Shallow Water Model
See fridom/framework/modules/__init__.py for more information.

## Available Modules:
    - NonlinearTendency: Nonlinear tendency term
    - NetCDFWriter: NetCDF writer module
    - Diagnostics: Diagnostic module
    - animation: Contains animation modules
    - advection: Contains advection modules
    - linear_tendency: Contains linear tendency modules
    - diffusion: Contains diffusion modules
"""

from .netcdf_writer import NetCDFWriter
from .diagnostics import Diagnostics

# move the animation module into the nonhydro namespace
from fridom.framework.modules import animation

from . import linear_tendency
from . import advection
from . import diffusion