"""Closures for the non-hydrostatic model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================

if TYPE_CHECKING:  # pragma: no cover
    from .diffusion import (
        BiharmonicFriction,
        BiharmonicMixing,
        HarmonicFriction,
        HarmonicMixing,
    )

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.shallowwater.modules.closures"
all_imports_by_origin = {
    f"{base}.diffusion": [
        "HarmonicMixing", "HarmonicFriction",
        "BiharmonicMixing", "BiharmonicFriction"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
