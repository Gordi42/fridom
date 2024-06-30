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
`BoundaryConditions`
    Boundary conditions module.
`LinearTendency`
    Linear tendency term.
`TendencyDivergence`
    Tendency divergence term (required for pressure solver).
`PressureGradientTendency`
    Pressure gradient tendency term.
`NetCDFWriter`
    NetCDF writer module.
`Diagnostics`
    Diagnostic module.

"""
# importing modules
# from . import advection
# from . import interpolation
# from . import diffusion
# from . import pressure_solvers
# from . import forcings

# importing classes
from .boundary_conditions import BoundaryConditions
from .linear_tendency import LinearTendency
# from .tendency_divergence import TendencyDivergence
from .pressure_gradient_tendency import PressureGradientTendency
# from .netcdf_writer import NetCDFWriter
# from .diagnostics import Diagnostics

# ----------------------------------------------------------------
#  Importing generic classes and modules
# ----------------------------------------------------------------
# importing modules
from fridom.framework.modules import animation

# importing classes