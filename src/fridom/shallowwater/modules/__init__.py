"""
A collection of modules for the shallow water model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # importing modules
    from . import closures

    # importing classes
    from .main_tendency import MainTendency
    from .linear_tendency import LinearTendency
    from .sadourny_advection import SadournyAdvection

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework.modules import animation

    # importing classes
    from fridom.framework.modules import NetCDFWriter, RestartModule, ResetTendency, FigureSaver

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
    f"{fr_base_path}": ["NetCDFWriter", 
                        "RestartModule", 
                        "ResetTendency", 
                        "FigureSaver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
