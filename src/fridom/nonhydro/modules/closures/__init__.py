"""Closures for the non-hydrostatic model."""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================

if TYPE_CHECKING:
    # import classes
    from .diffusion import HarmonicMixing, HarmonicFriction, \
        BiharmonicMixing, BiharmonicFriction
    from .smagorinsky_lilly import SmagorinskyLilly

    # import modules


# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.nonhydro.modules.closures"
all_imports_by_origin = {
    f"{base}.diffusion": [
        "HarmonicMixing", "HarmonicFriction",
        "BiharmonicMixing", "BiharmonicFriction"],
    f"{base}.smagorinsky_lilly": ["SmagorinskyLilly"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
