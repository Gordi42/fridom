"""
# Model Modules for the Shallow Water Model
See fridom/framework/modules/__init__.py for more information.

## Available Modules:
    - LinearTendency: Linear tendency term
    - NonlinearTendency: Nonlinear tendency term
"""

from .linear_tendency import LinearTendency, LinearTendencyFD, LinearTendencySpectral
from .nonlinear_tendency import NonlinearTendency, NonlinearTendencyFD, NonlinearTendencySpectral
from .netcdf_writer import NetCDFWriter