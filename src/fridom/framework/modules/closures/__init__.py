"""
Closure modules. E.g. diffusion, hyperdiffusion, etc.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import classes
    from .harmonic_diffusion import HarmonicDiffusion
    from .biharmonic_diffusion import BiharmonicDiffusion

    # import modules


# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.framework.modules.closures"
all_imports_by_origin = {
    f"{base}.harmonic_diffusion": ["HarmonicDiffusion"],
    f"{base}.biharmonic_diffusion": ["BiharmonicDiffusion"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)