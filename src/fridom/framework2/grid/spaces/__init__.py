"""
Function-space families of the grid cluster.

Description
-----------
Owning class docs: ``notes/framework2/classes/spaces.md`` and
``notes/framework2/classes/product_spaces.md``. Spaces get no
top-level namespace (they are produced by mesh factories);
these modules are importable for ``isinstance`` checks in
operator/dispatch code, not for construction.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        average,
        coefficient,
        constant,
        function_space,
        galerkin,
        nodal,
        tensor_product,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.spaces"

all_modules_by_origin = {
    base: [
        "function_space",
        "nodal",
        "average",
        "coefficient",
        "galerkin",
        "constant",
        "tensor_product",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
