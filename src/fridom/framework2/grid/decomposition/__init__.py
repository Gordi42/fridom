"""
Per-mesh domain-decomposition layer.

Description
-----------
Owning class doc:
``notes/framework2/classes/decomposition.md``. Public for
transform/solver authors as ``fr.grid.decomposition``; not
re-exported at ``fr.*`` level. Waves 1/3 re-export the
decomposition classes here.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        decomposition,
        graph,
        halo,
        layout,
        tensor,
        traits,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.decomposition"

all_modules_by_origin = {
    base: [
        "traits",
        "halo",
        "layout",
        "decomposition",
        "tensor",
        "graph",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
