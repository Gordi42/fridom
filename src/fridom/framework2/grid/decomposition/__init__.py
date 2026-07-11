"""
Per-mesh domain-decomposition layer.

Description
-----------
Owning class doc:
``design/specs/grid/classes/decomposition.md``. Public for
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

    # import all classes and functions
    from .decomposition import (
        Decomposition,
        ReshardingReport,
        SpaceLike,
        negotiate,
    )
    from .halo import HaloSpec, HaloTracer, trace_halo
    from .layout import Layout
    from .tensor import TensorDecomposition
    from .traits import HaloStrategy, MeshDecompositionTraits

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

all_imports_by_origin = {
    f"{base}.traits": ["HaloStrategy", "MeshDecompositionTraits"],
    f"{base}.halo": ["HaloSpec", "HaloTracer", "trace_halo"],
    f"{base}.layout": ["Layout"],
    f"{base}.decomposition": [
        "Decomposition",
        "ReshardingReport",
        "SpaceLike",
        "negotiate",
    ],
    f"{base}.tensor": ["TensorDecomposition"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
