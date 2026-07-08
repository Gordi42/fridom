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
        banded,
        base,
        block_symbol,
        chebyshev,
        combinators,
        composed,
        dealias,
        finite_difference,
        flux_diff,
        fourier,
        interp,
        movement,
        products,
        reconstruct,
        registry,
        select,
        spectral,
        spectral_solve,
        symbol,
        transform,
        trig,
        verbs,
        weno,
    )

    # import all classes and objects
    from .base import (
        BinaryOperator,
        Block,
        Composite,
        Dispatched,
        EigenbasisError,
        Identity,
        Operator,
        OperatorRequirements,
        OperatorSum,
        ScaledOperator,
        SeparableComposite,
        SeparableOperator,
        UnaryOperator,
        Zero,
        resolve_codomain,
    )
    from .block_symbol import BlockSymbol
    from .composed import Curl, Divergence, Gradient, Laplacian
    from .finite_difference import FiniteDifference
    from .flux_diff import (
        DualFluxDifference,
        FaceDifference,
        FluxDifference,
        FVDerivative,
    )
    from .integrate import Integral
    from .interp import LinearInterp
    from .movement import Reshard, Sync
    from .reconstruct import LinearReconstruction
    from .registry import DispatchError, DispatchKey, OperatorRegistry
    from .select import Where
    from .spectral_solve import SpectralSolve
    from .symbol import Symbol
    from .verbs import diff, integrate, interpolate
    from .weno import WenoReconstruction

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.operators"

all_modules_by_origin = {
    base: [
        "base",
        "registry",
        "banded",
        "symbol",
        "block_symbol",
        "spectral_solve",
        "finite_difference",
        "interp",
        "reconstruct",
        "weno",
        "select",
        "flux_diff",
        "spectral",
        "transform",
        "fourier",
        "trig",
        "chebyshev",
        "products",
        # the "integrate" module is NOT re-exported by name: the
        # D3b verb below owns the ``fr.operators.integrate`` slot
        # (the module stays importable by its full path).
        "composed",
        "dealias",
        "combinators",
        "movement",
        "verbs",
    ],
}

all_imports_by_origin = {
    f"{base}.base": [
        "Operator",
        "UnaryOperator",
        "BinaryOperator",
        "SeparableOperator",
        "OperatorRequirements",
        "EigenbasisError",
        "Identity",
        "Zero",
        "Composite",
        "SeparableComposite",
        "OperatorSum",
        "ScaledOperator",
        "Block",
        "Dispatched",
        "resolve_codomain",
    ],
    f"{base}.registry": [
        "OperatorRegistry",
        "DispatchError",
        "DispatchKey",
    ],
    f"{base}.finite_difference": ["FiniteDifference"],
    f"{base}.interp": ["LinearInterp"],
    f"{base}.reconstruct": ["LinearReconstruction"],
    f"{base}.weno": ["WenoReconstruction"],
    f"{base}.select": ["Where"],
    f"{base}.spectral_solve": ["SpectralSolve"],
    f"{base}.symbol": ["Symbol"],
    f"{base}.block_symbol": ["BlockSymbol"],
    f"{base}.flux_diff": [
        "FluxDifference",
        "DualFluxDifference",
        "FaceDifference",
        "FVDerivative",
    ],
    # the class import "Integral" and the D3b verb "integrate"
    # coexist: only the verb owns the ``fr.operators.integrate`` slot
    f"{base}.integrate": ["Integral"],
    f"{base}.composed": [
        "Gradient",
        "Divergence",
        "Curl",
        "Laplacian",
    ],
    f"{base}.movement": ["Reshard", "Sync"],
    f"{base}.verbs": ["diff", "interpolate", "integrate"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
