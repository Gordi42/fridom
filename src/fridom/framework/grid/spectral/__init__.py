
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import modules

    # import classes
    from .grid import Grid
    from .fft import FFT
    from .spectral_diff import SpectralDiff
    
# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid.cartesian"

all_modules_by_origin = {
}

all_imports_by_origin = { 
    f"{base}.grid": ["Grid"],
    f"{base}.fft": ["FFT"],
    f"{base}.spectral_diff": ["SpectralDiff"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)