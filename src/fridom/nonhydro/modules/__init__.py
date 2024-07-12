"""
Nonhydrostatic Model Modules
============================
A collection of modules for the nonhydrostatic model.

Modules
-------
`animation`
    Modules for creating animated output of the model.
`advection`
    Modules for advection schemes.
`interpolation`
    Modules for interpolation schemes.
`diffusion`
    Modules for diffusion schemes.
`pressure_solvers`
    Modules for pressure solver schemes.
`forcings`
    Modules for forcing schemes.

Classes
-------
`RestartModule`
    A module for restarting the model.
`BoundaryConditions`
    Boundary conditions module.
`LinearTendency`
    Linear tendency term.
`TendencyDivergence`
    Tendency divergence term (required for pressure solver).
`PressureGradientTendency`
    Pressure gradient tendency term.
`MainTendency`
    Main tendency module.
`NetCDFWriter`
    NetCDF writer module (for parallel output).
`NetCDFWriterSingleProc`
    NetCDF writer module for single processor.
`Diagnostics`
    Diagnostic module.

"""
# importing modules
# from . import advection
# from . import interpolation
# from . import diffusion
from . import pressure_solvers
# from . import forcings

# importing classes
from .boundary_conditions import BoundaryConditions
from .linear_tendency import LinearTendency
from .tendency_divergence import TendencyDivergence
from .pressure_gradient_tendency import PressureGradientTendency
from .main_tendency import MainTendency
from .netcdf_writer_single_proc import NetCDFWriterSingleProc
# from .diagnostics import Diagnostics

# ----------------------------------------------------------------
#  Importing generic classes and modules
# ----------------------------------------------------------------
# importing modules
from fridom.framework.modules import animation

# importing classes
from fridom.framework.modules.netcdf_writer import NetCDFWriter
from fridom.framework.modules.restart_module import RestartModule