"""
Function-space families of the grid cluster.

Description
-----------
Owning class docs: ``design/specs/grid/classes/spaces.md`` and
``design/specs/grid/classes/product_spaces.md``. Spaces get no
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
        composition,
        constant,
        function_space,
        galerkin,
        nodal,
        tensor_product,
        trace,
    )

    # import all functions
    from .composition import compose_spaces, union_spaces
    from .trace import Side, TraceSpace

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial.spaces"

all_modules_by_origin = {
    base: [
        "function_space",
        "nodal",
        "average",
        "coefficient",
        "galerkin",
        "constant",
        "tensor_product",
        "composition",
        "trace",
    ],
}

all_imports_by_origin = {
    f"{base}.composition": ["compose_spaces", "union_spaces"],
    f"{base}.trace": ["Side", "TraceSpace"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
