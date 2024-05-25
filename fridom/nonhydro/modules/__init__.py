"""
# Model Modules for the Nonhydrostatic Model
See fridom/framework/modules/__init__.py for more information.

## Available Modules:
    - BiharmonicFriction: Tendency due to biharmonic friction
    - BiharmonicMixing: Tendency due to biharmonic mixing
    - HarmonicFriction: Tendency due to harmonic friction
    - HarmonicMixing: Tendency due to harmonic mixing
    - LinearTendency: Linear tendency term
    - PressureGradientTendency: Pressure gradient tendency term
    - PressureSolve: Pressure solver
    - SourceTendency: Source term tendency
    - NetCDFWriter: NetCDF writer module
    - animation: Contains animation modules
    - advection: Contains advection modules
    - interpolation: Contains interpolation modules
"""

from .biharmonic_friction import BiharmonicFriction
from .biharmonic_mixing import BiharmonicMixing
from .harmonic_friction import HarmonicFriction
from .harmonic_mixing import HarmonicMixing
from .linear_tendency import LinearTendency
from .pressure_gradient_tendency import PressureGradientTendency
from .pressure_solve import PressureSolve
from .source_tendency import SourceTendency
from .netcdf_writer import NetCDFWriter

# move the animation module into the nonhydro namespace
from fridom.framework.modules import animation

from . import advection
from . import interpolation