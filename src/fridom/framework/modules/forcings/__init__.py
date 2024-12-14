"""Forcing modules. E.g. relaxation."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .relaxation import Relaxation

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

base = "fridom.framework.modules.forcings"
all_imports_by_origin = {
    f"{base}.relaxation": ["Relaxation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
