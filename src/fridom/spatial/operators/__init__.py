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
        cumulative,
        dealias,
        distributed_solve,
        finite_difference,
        flux_diff,
        fourier,
        interp,
        krylov,
        mixed,
        movement,
        multigrid,
        products,
        realized,
        reconstruct,
        registry,
        restrict,
        select,
        spectral,
        spectral_solve,
        symbol,
        transfer,
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
    from .chebyshev import Chebyshev
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
    from .cumulative import CumulativeIntegral
    from .distributed_solve import resolve_distributed_solve
    from .finite_difference import FiniteDifference
    from .flux_diff import (
        DualFluxDifference,
        FaceDifference,
        FluxDifference,
        FVDerivative,
    )
    from .fourier import Fourier
    from .integrate import Integral
    from .interp import LinearInterp
    from .krylov import ConjugateGradient
    from .mapped import MappedDerivative, MetricScaled
    from .mixed import ComposedTransform, resolve_transform
    from .movement import Reshard, Sync
    from .multigrid import (
        DampedJacobi,
        MultigridLevel,
        MultigridVCycle,
        VerticalLineJacobi,
    )
    from .realized import (
        BoundTransform,
        RealizedComposite,
        RealizedMap,
        RealizedSum,
    )
    from .reconstruct import LinearDeconvolution, LinearReconstruction
    from .registry import DispatchError, DispatchKey, OperatorRegistry
    from .restrict import Restriction
    from .select import Where
    from .spectral import PhaseShift, SpectralDerivative
    from .spectral_solve import SpectralSolve
    from .symbol import Symbol
    from .transfer import GridTransfer
    from .trig import Cosine, Sine
    from .verbs import (
        cumint,
        diff,
        integrate,
        interpolate,
        physical_diff,
    )
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
        "restrict",
        "select",
        "flux_diff",
        "cumulative",
        "spectral",
        "transfer",
        "transform",
        "mixed",
        "mapped",
        "fourier",
        "trig",
        "chebyshev",
        "multigrid",
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
    f"{base}.reconstruct": [
        "LinearDeconvolution", "LinearReconstruction"],
    f"{base}.weno": ["WenoReconstruction"],
    f"{base}.restrict": ["Restriction"],
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
    # transform classes (were reachable only via their leaf modules)
    f"{base}.fourier": ["Fourier"],
    f"{base}.trig": ["Sine", "Cosine"],
    f"{base}.chebyshev": ["Chebyshev"],
    f"{base}.spectral": ["SpectralDerivative", "PhaseShift"],
    f"{base}.transfer": ["GridTransfer"],
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
    f"{base}.multigrid": [
        "MultigridVCycle",
        "MultigridLevel",
        "DampedJacobi",
        "VerticalLineJacobi",
    ],
    f"{base}.cumulative": ["CumulativeIntegral"],
    f"{base}.verbs": ["diff", "interpolate", "integrate", "cumint",
                      "physical_diff"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
