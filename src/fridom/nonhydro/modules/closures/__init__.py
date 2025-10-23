"""Closures for the non-hydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================

if TYPE_CHECKING:  # pragma: no cover
    from .biharmonic_closure import BiharmonicClosure
    from .diffusion import (
        BiharmonicFriction,
        BiharmonicMixing,
        HarmonicFriction,
        HarmonicMixing,
    )
    from .smagorinsky_lilly import SmagorinskyLilly

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.nonhydro.modules.closures"
all_imports_by_origin = {
    f"{base}.biharmonic_closure": ["BiharmonicClosure"],
    f"{base}.diffusion": [
        "HarmonicMixing", "HarmonicFriction",
        "BiharmonicMixing", "BiharmonicFriction"],
    f"{base}.smagorinsky_lilly": ["SmagorinskyLilly"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
