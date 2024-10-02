"""
Initial Conditions for the shallow water model
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .jet import Jet
    from .single_wave import SingleWave
    from .coherent_eddy import CoherentEddy
    
# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.shallowwater.initial_conditions"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.jet": ["Jet"],
    f"{base_path}.coherent_eddy": ["CoherentEddy"],
    f"{base_path}.single_wave": ["SingleWave"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
