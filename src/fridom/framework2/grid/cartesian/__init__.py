"""
Cartesian convenience grid.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``.
Wave 2 re-exports the cartesian ``Grid`` class here.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        grid,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.cartesian"

all_modules_by_origin = {
    base: [
        "grid",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
