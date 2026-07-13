"""
Free-standing operators (re-exported as ``fr.operators``).

Description
-----------
Owning class docs: the
``design/specs/grid/classes/operators_*.md`` cluster files.
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
        chebyshev,
        combinators,
        composed,
        dealias,
        distributed_solve,
        finite_difference,
        flux_diff,
        fourier,
        interp,
        krylov,
        mixed,
        movement,
        products,
        realized,
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
    from .composed import (
        Curl,
        Diag,
        Divergence,
        Gradient,
        Laplacian,
        LowerIndex,
        MetricCurl,
        MetricDivergence,
        MetricGradient,
        MetricLaplacian,
        RaiseIndex,
        VarianceRetag,
    )
    from .distributed_solve import resolve_distributed_solve
    from .finite_difference import FiniteDifference
    from .flux_diff import (
        DualFluxDifference,
        FaceDifference,
        FluxDifference,
        FVDerivative,
    )
    from .integrate import Integral
    from .interp import LinearInterp
    from .krylov import ConjugateGradient
    from .mapped import MappedDerivative, MetricScaled
    from .mixed import ComposedTransform, resolve_transform
    from .movement import Reshard, Sync
    from .realized import (
        BoundTransform,
        RealizedComposite,
        RealizedMap,
        RealizedSum,
    )
    from .reconstruct import LinearReconstruction
    from .registry import DispatchError, DispatchKey, OperatorRegistry
    from .select import Where
    from .spectral_solve import SpectralSolve
    from .symbol import Symbol
    from .verbs import diff, integrate, interpolate, physical_diff
    from .weno import WenoReconstruction

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial.operators"

all_modules_by_origin = {
    base: [
        "base",
        "registry",
        "banded",
        "symbol",
        "realized",
        "distributed_solve",
        "spectral_solve",
        "finite_difference",
        "interp",
        "krylov",
        "reconstruct",
        "weno",
        "select",
        "flux_diff",
        "spectral",
        "transform",
        "mixed",
        "mapped",
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
    f"{base}.krylov": ["ConjugateGradient"],
    f"{base}.reconstruct": ["LinearReconstruction"],
    f"{base}.weno": ["WenoReconstruction"],
    f"{base}.select": ["Where"],
    f"{base}.distributed_solve": ["resolve_distributed_solve"],
    f"{base}.spectral_solve": ["SpectralSolve"],
    f"{base}.realized": [
        "RealizedMap",
        "RealizedComposite",
        "RealizedSum",
        "BoundTransform",
    ],
    f"{base}.symbol": ["Symbol"],
    f"{base}.flux_diff": [
        "FluxDifference",
        "DualFluxDifference",
        "FaceDifference",
        "FVDerivative",
    ],
    # the class import "Integral" and the D3b verb "integrate"
    # coexist: only the verb owns the ``fr.operators.integrate`` slot
    f"{base}.integrate": ["Integral"],
    f"{base}.mapped": ["MappedDerivative", "MetricScaled"],
    f"{base}.mixed": ["ComposedTransform", "resolve_transform"],
    f"{base}.composed": [
        "Gradient",
        "Divergence",
        "Curl",
        "Diag",
        "Laplacian",
        "MetricGradient",
        "MetricDivergence",
        "MetricCurl",
        "MetricLaplacian",
        "RaiseIndex",
        "LowerIndex",
        "VarianceRetag",
    ],
    f"{base}.movement": ["Reshard", "Sync"],
    f"{base}.verbs": ["diff", "interpolate", "integrate",
                      "physical_diff"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
