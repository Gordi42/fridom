"""A collection of modules for the Hydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.modules import (
        FigureSaver,
        Module,
        NaNChecker,
        NetCDFWriter,
        Ramper,
        ResetTendency,
        RestartModule,
        animation,
        module_method,
    )

    from .main_tendency import MainTendency

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.hydrostatic.modules"
fr_base_path = "fridom.framework.modules"

all_modules_by_origin = {
    fr_base_path: ["animation"],
}

all_imports_by_origin = {
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{fr_base_path}": ["NaNChecker",
                        "NetCDFWriter",
                        "Ramper",
                        "RestartModule",
                        "ResetTendency"],
    f"{fr_base_path}.module": ["Module", "module_method"],
    f"{fr_base_path}.figure_saver": ["FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
