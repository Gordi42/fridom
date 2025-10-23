"""Closure modules. E.g. diffusion, hyperdiffusion, etc."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .biharmonic_closure import BiharmonicClosure
    from .biharmonic_diffusion import BiharmonicDiffusion
    from .harmonic_diffusion import HarmonicDiffusion

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.framework.modules.closures"
all_imports_by_origin = {
    f"{base}.biharmonic_diffusion": ["BiharmonicDiffusion"],
    f"{base}.harmonic_diffusion": ["HarmonicDiffusion"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
