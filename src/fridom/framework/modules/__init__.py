"""Base classes and functions for creating modules in the model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # importing modules
    from . import advection, animation, closures, forcings
    from .figure_saver import FigureSaver

    # importing the classes and functions
    from .module import Module, module_method
    from .module_container import ModuleContainer
    from .nan_checker import NaNChecker
    from .netcdf_writer import NetCDFWriter
    from .progress_bar import ProgressBar
    from .reset_tendency import ResetTendency
    from .restart_module import RestartModule

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
    f"{base_path}.nan_checker": ["NaNChecker"],
    f"{base_path}.netcdf_writer": ["NetCDFWriter"],
    f"{base_path}.reset_tendency": ["ResetTendency"],
    f"{base_path}.figure_saver": ["FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
