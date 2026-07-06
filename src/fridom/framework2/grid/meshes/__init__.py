"""
Mesh factors of the domain (re-exported as ``fr.meshes``).

Description
-----------
Owning class doc: ``notes/framework2/classes/meshes.md``.
Wave 1 re-exports the mesh classes (``IntervalMesh``, ...)
here.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        chebyshev,
        interval,
        mapped_interval,
        mesh,
        point,
        sphere,
        structured_1d,
        unstructured,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.meshes"

all_modules_by_origin = {
    base: [
        "mesh",
        "structured_1d",
        "interval",
        "point",
        "mapped_interval",
        "chebyshev",
        "sphere",
        "unstructured",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
