"""
Spherical convenience grid.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``. Re-exports the
lat-lon sphere convenience ``Grid`` — two ``IntervalMesh`` factors under
the orthogonal sphere chart.
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

    # import all classes
    from .grid import Grid

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial.spherical"

all_modules_by_origin = {
    base: [
        "grid",
    ],
}

all_imports_by_origin = {
    f"{base}.grid": ["Grid"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
