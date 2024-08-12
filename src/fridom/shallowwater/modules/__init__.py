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
    # from . import advection

    # importing classes
    from .main_tendency import MainTendency
    from .linear_tendency import LinearTendency

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework.modules import animation

    # importing classes
    from fridom.framework.modules import NetCDFWriter, RestartModule, ResetTendency

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.shallowwater.modules"
fr_base_path = "fridom.framework.modules"

all_modules_by_origin = { 
    fr_base_path: ["animation"],
}

all_imports_by_origin = { 
    f"{base_path}.main_tendency": ["MainTendency"],
    f"{base_path}.linear_tendency": ["LinearTendency"],
    f"{fr_base_path}": ["NetCDFWriter", "RestartModule", "ResetTendency"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
