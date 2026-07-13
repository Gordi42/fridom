"""
Mesh factors of the domain (re-exported as ``fr.meshes``).

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md``.
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

    # import all classes
    from .chebyshev import ChebyshevMesh
    from .interval import IntervalMesh
    from .mapped_interval import MappedIntervalMesh
    from .mesh import Mesh
    from .point import PointMesh
    from .structured_1d import StructuredMesh1D

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial.meshes"

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

all_imports_by_origin = {
    f"{base}.mesh": ["Mesh"],
    f"{base}.structured_1d": ["StructuredMesh1D"],
    f"{base}.interval": ["IntervalMesh"],
    f"{base}.mapped_interval": ["MappedIntervalMesh"],
    f"{base}.point": ["PointMesh"],
    f"{base}.chebyshev": ["ChebyshevMesh"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
