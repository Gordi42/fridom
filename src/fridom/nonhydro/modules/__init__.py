"""A collection of modules for the nonhydrostatic model."""
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
        SyncModule,
        ZarrWriter,
        animation,
        module_method,
    )

    from . import advection, closures, forcings, pressure_solvers
    from .linear_tendency import LinearTendency
    from .main_tendency import MainTendency
    from .pressure_gradient_tendency import PressureGradientTendency
    from .tendency_divergence import TendencyDivergence

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
        "advection",
        "forcings"],
    fr_base_path: ["animation"],
}

all_imports_by_origin = {
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{base_path}.tendency_divergence": ["TendencyDivergence"],
    f"{base_path}.pressure_gradient_tendency": ["PressureGradientTendency"],
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{fr_base_path}": ["NaNChecker",
                        "NetCDFWriter",
                        "Ramper",
                        "RestartModule",
                        "ResetTendency",
                        "SyncModule",
                        "ZarrWriter"],
    f"{fr_base_path}.module": ["Module", "module_method"],
    f"{fr_base_path}.figure_saver": ["FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
