"""
This module contains the diffusion modules (friction and mixing) of the model.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .harmonic_friction import HarmonicFriction
    from .harmonic_mixing import HarmonicMixing
    from .biharmonic_friction import BiharmonicFriction
    from .biharmonic_mixing import BiharmonicMixing

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.nonhydro.modules.diffusion"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.harmonic_friction": ["HarmonicFriction"],
    f"{base_path}.harmonic_mixing": ["HarmonicMixing"],
    f"{base_path}.biharmonic_friction": ["BiharmonicFriction"],
    f"{base_path}.biharmonic_mixing": ["BiharmonicMixing"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
