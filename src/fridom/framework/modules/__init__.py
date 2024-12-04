"""
Base classes and functions for creating modules in the model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # importing modules
    from . import advection
    from . import animation
    from . import closures
    from . import forcings

    # importing the Classes
    from .module import Module
    from .module_container import ModuleContainer
    from .progress_bar import ProgressBar
    from .restart_module import RestartModule
    from .netcdf_writer import NetCDFWriter
    from .reset_tendency import ResetTendency
    from .figure_saver import FigureSaver

    # importing the functions
    from .module import module_method

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.modules"

all_modules_by_origin = { base_path: ["animation", 
                                      "advection",
                                      "closures",
                                      "forcings"] }

all_imports_by_origin = { 
    f"{base_path}.module": ["Module", "module_method"],
    f"{base_path}.module_container": ["ModuleContainer"],
    f"{base_path}.progress_bar": ["ProgressBar"],
    f"{base_path}.restart_module": ["RestartModule"],
    f"{base_path}.netcdf_writer": ["NetCDFWriter"],
    f"{base_path}.reset_tendency": ["ResetTendency"],
    f"{base_path}.figure_saver": ["FigureSaver"]
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
