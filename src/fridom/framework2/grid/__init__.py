"""
Grid abstraction cluster of framework2.

Description
-----------
The re-export table below realizes section 1 of
``notes/framework2/classes/grid.md``; at cutover the same entries
move up to the framework ``__init__``. Entries whose modules are
still stubs are added wave by wave (see the wave markers in the
modules):

- Wave 1 adds ``TensorProductSpace`` and ``SpaceLike`` from
  ``spaces.tensor_product``.
- Wave 2 adds ``Grid`` from ``grid`` and ``ScalarField`` /
  ``FieldMetadata`` from ``fields``.
- Wave 3 adds ``VectorField`` / ``TensorField`` from ``fields``.
- Wave 4 adds ``ImmersedDomain`` / ``Slip`` from ``immersed_domain``
  and ``CoordinateMapping`` from ``coordinate_mapping``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import cartesian, decomposition, meshes, operators

    # import all classes
    from .bc import BC
    from .errors import GridMismatchError, SpaceMismatchError
    from .fields import FieldMetadata, ScalarField, VectorField
    from .grid import Grid
    from .immersed_domain import ImmersedDomain, Slip
    from .scalars import Complex, Real
    from .spaces.tensor_product import SpaceLike, TensorProductSpace

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid"

all_modules_by_origin = {
    base: ["meshes", "operators", "cartesian", "decomposition"],
}

all_imports_by_origin = {
    f"{base}.grid": ["Grid"],
    f"{base}.scalars": ["Real", "Complex"],
    f"{base}.bc": ["BC"],
    f"{base}.errors": ["SpaceMismatchError", "GridMismatchError"],
    f"{base}.spaces.tensor_product": ["TensorProductSpace",
                                      "SpaceLike"],
    f"{base}.fields": ["ScalarField", "FieldMetadata",
                       "VectorField"],
    f"{base}.immersed_domain": ["ImmersedDomain", "Slip"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
