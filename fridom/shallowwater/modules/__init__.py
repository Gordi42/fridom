"""
# Model Modules for the Shallow Water Model
See fridom/framework/modules/__init__.py for more information.

## Available Modules:
    - LinearTendency: Linear tendency term
    - NonlinearTendency: Nonlinear tendency term
    - NetCDFWriter: NetCDF writer module
    - animation: Contains animation modules
"""

from .linear_tendency import LinearTendency, LinearTendencyFD, LinearTendencySpectral
from .nonlinear_tendency import NonlinearTendency, NonlinearTendencyFD, NonlinearTendencySpectral
from .netcdf_writer import NetCDFWriter

# move the animation module into the nonhydro namespace
from fridom.framework.modules import animation