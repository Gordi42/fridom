"""
A collection of modules for the nonhydrostatic model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # importing modules
    from . import advection
    from . import diffusion
    from . import pressure_solvers
    from . import forcings

    # importing classes
    from .boundary_conditions import BoundaryConditions
    from .linear_tendency import LinearTendency
    from .tendency_divergence import TendencyDivergence
    from .pressure_gradient_tendency import PressureGradientTendency
    from .main_tendency import MainTendency
    from .netcdf_writer_single_proc import NetCDFWriterSingleProc
    from .diagnostics import Diagnostics

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework.modules import animation

    # importing classes
    from fridom.framework.modules.netcdf_writer import NetCDFWriter
    from fridom.framework.modules.restart_module import RestartModule
    from fridom.framework.modules.reset_tendency import ResetTendency

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules"
fr_base_path = "fridom.framework.modules"

all_modules_by_origin = { 
    base_path: [
        "advection", 
        "diffusion", 
        "pressure_solvers", 
        "forcings"], 
    fr_base_path: ["animation"],
}

all_imports_by_origin = { 
    f"{base_path}.boundary_conditions": ["BoundaryConditions"],
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{base_path}.tendency_divergence": ["TendencyDivergence"],
    f"{base_path}.pressure_gradient_tendency": ["PressureGradientTendency"],
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{base_path}.netcdf_writer_single_proc": ["NetCDFWriterSingleProc"],
    f"{base_path}.diagnostics": ["Diagnostics"],
    f"{fr_base_path}.netcdf_writer": ["NetCDFWriter"],
    f"{fr_base_path}.restart_module": ["RestartModule"], 
    f"{fr_base_path}.reset_tendency": ["ResetTendency"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
