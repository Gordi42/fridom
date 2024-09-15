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
    # from . import advection
    from . import closures
    from . import pressure_solvers
    from . import forcings

    # importing classes
    from .linear_tendency import LinearTendency
    from .tendency_divergence import TendencyDivergence
    from .pressure_gradient_tendency import PressureGradientTendency
    from .main_tendency import MainTendency
    from .diagnostics import Diagnostics

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework.modules import animation
    from fridom.framework.modules import advection

    # importing classes
    from fridom.framework.modules import Module
    from fridom.framework.modules import NetCDFWriter, RestartModule, ResetTendency
    from fridom.framework.modules import FigureSaver

    # importing the functions
    from fridom.framework.modules import module_method

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules"
fr_base_path = "fridom.framework.modules"

all_modules_by_origin = { 
    base_path: [
        "advection", 
        "closures",
        "pressure_solvers", 
        "forcings"], 
    fr_base_path: ["animation", "advection"],
}

all_imports_by_origin = { 
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{base_path}.tendency_divergence": ["TendencyDivergence"],
    f"{base_path}.pressure_gradient_tendency": ["PressureGradientTendency"],
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{base_path}.diagnostics": ["Diagnostics"],
    f"{fr_base_path}": ["NetCDFWriter", "RestartModule", "ResetTendency"],
    f"{fr_base_path}.module": ["Module", "module_method"],
    f"{fr_base_path}.figure_saver": ["FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
