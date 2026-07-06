"""
Free-standing operators (re-exported as ``fr.operators``).

Description
-----------
Owning class docs: the
``notes/framework2/classes/operators_*.md`` cluster files.
Waves 2-4 re-export the operator classes here.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        base,
        chebyshev,
        combinators,
        composed,
        dealias,
        finite_difference,
        flux_diff,
        fourier,
        integrate,
        interp,
        movement,
        products,
        reconstruct,
        registry,
        spectral,
        symbol,
        transform,
        trig,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.operators"

all_modules_by_origin = {
    base: [
        "base",
        "registry",
        "symbol",
        "finite_difference",
        "interp",
        "reconstruct",
        "flux_diff",
        "spectral",
        "transform",
        "fourier",
        "trig",
        "chebyshev",
        "products",
        "integrate",
        "composed",
        "dealias",
        "combinators",
        "movement",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
