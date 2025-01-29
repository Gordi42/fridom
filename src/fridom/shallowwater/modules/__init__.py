"""A collection of modules for the shallow water model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework.modules import (
        FigureSaver,
        NaNChecker,
        NetCDFWriter,
        Ramper,
        ResetTendency,
        RestartModule,
        animation,
    )

    from . import closures
    from .linear_tendency import LinearTendency
    from .main_tendency import MainTendency
    from .sadourny_advection import SadournyAdvection

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.shallowwater.modules"
fr_base_path = "fridom.framework.modules"

all_modules_by_origin = {
    base_path: ["closures"],
    fr_base_path: ["animation"],
}

all_imports_by_origin = {
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{base_path}.sadourny_advection": ["SadournyAdvection"],
    f"{fr_base_path}": ["NaNChecker",
                        "NetCDFWriter",
                        "Ramper",
                        "RestartModule",
                        "ResetTendency",
                        "FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
