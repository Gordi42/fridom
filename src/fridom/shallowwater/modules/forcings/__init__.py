"""Forcing modules for the shallow water model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================

if TYPE_CHECKING:  # pragma: no cover
    from .gaussian_wave_maker import GaussianWaveMaker
    from .polarized_wave_maker import PolarizedWaveMaker

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
