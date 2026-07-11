"""
``fridom.spatial`` — spatial discretization (meshes, spaces, fields).

Description
-----------
Everything about discretizing space: meshes, function spaces, fields,
the shared operator numerics library, domain decomposition, the
``Grid`` assembly root, and the declaration-tag vocabulary
(``space_patterns``). The concrete models (``fridom.nonhydro2``,
``fridom.shallowwater2``) and the temporal layer (``fridom.model``)
consume this package; it depends on none of them.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        cartesian,
        decomposition,
        meshes,
        operators,
        space_patterns,
    )

    # import all classes
    from .bc import BC
    from .errors import GridMismatchError, SpaceMismatchError
    from .fields import FieldMetadata, ScalarField, VectorField
    from .grid import Grid
    from .immersed_domain import ImmersedDomain, Slip
    from .scalars import Complex, Real
    from .space_patterns import (
        Collocated,
        Dof,
        Profile,
        SpacePattern,
        SpaceRule,
        Staggered,
    )
    from .spaces.tensor_product import SpaceLike, TensorProductSpace
    from .symbols import GridSymbols, ModeChart, rayleigh_dual

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial"

all_modules_by_origin = {
    base: [
        "meshes", "operators", "cartesian", "decomposition",
        "space_patterns"],
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
    f"{base}.symbols": ["GridSymbols", "ModeChart", "rayleigh_dual"],
    f"{base}.space_patterns": [
        "Dof", "SpacePattern", "Collocated", "Staggered", "Profile",
        "SpaceRule"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
