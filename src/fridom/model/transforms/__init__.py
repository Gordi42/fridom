"""
The state-transform namespace (``fr.transforms``).

Description
-----------
Owning class spec: ``design/specs/model/classes/transforms.md``;
design source ``design/specs/model/08_state_transforms.md``
(§10.1-10.8). Wave 7 A populates the base + algebra: the
``StateTransform`` base (re-exported top-level as ``fr.StateTransform``),
the signature/info/cost/progress vocabulary, the error types, the
algebra nodes, ``Identity``/``Shift``/``FixedPoint``, and
``relative_l2``/``assert_idempotent``. The Tier-2 presets
(``Propagator``, ``TimeAverage``, ``OptimalBalance``, wave 7 B) and
the eigenmode projections (wave 7 C) land next. Spectral *operator*
transforms stay in the grid layer; this is the ``State -> State``
namespace.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        adiabatic_projection,
        adiabatic_ramping,
        algebra,
        balance_expansion,
        base,
        errors,
        fixed_point,
        identity,
        info,
        norms,
        optimal_balance,
        projection,
        propagator,
        shift,
        signature,
        time_average,
    )

    # import all classes
    from .adiabatic_projection import AdiabaticProjection
    from .adiabatic_ramping import AdiabaticRamping
    from .algebra import Compose, Power, Scaled, Sum
    from .balance_expansion import BalanceExpansion
    from .base import StateTransform
    from .errors import (
        FixedPointDivergenceError,
        SignatureMismatchError,
        TraceError,
    )
    from .fixed_point import FixedPoint
    from .identity import Identity
    from .info import TransformCost, TransformInfo, TransformProgress
    from .norms import assert_idempotent, relative_imbalance, relative_l2
    from .optimal_balance import OptimalBalance
    from .projection import (
        EigenFunction,
        EigenProjection,
        ProjectionFactory,
    )
    from .propagator import Propagator
    from .shift import Shift
    from .signature import StateSignature
    from .time_average import TimeAverage

# ================================================================
#  Setup lazy loading
# ================================================================
# NB: the package's own ``base`` submodule (the StateTransform base)
# would shadow the conventional ``base = "..."`` origin variable, so
# the origin path is spelled ``pkg`` here.
pkg = "fridom.model.transforms"

all_modules_by_origin = {
    pkg: [
        "signature",
        "info",
        "errors",
        "base",
        "algebra",
        "identity",
        "shift",
        "fixed_point",
        "norms",
        "propagator",
        "time_average",
        "adiabatic_ramping",
        "adiabatic_projection",
        "optimal_balance",
        "projection",
        "balance_expansion",
    ],
}

all_imports_by_origin = {
    f"{pkg}.base": ["StateTransform"],
    f"{pkg}.signature": ["StateSignature"],
    f"{pkg}.info": [
        "TransformInfo", "TransformCost", "TransformProgress"],
    f"{pkg}.errors": [
        "SignatureMismatchError", "TraceError",
        "FixedPointDivergenceError"],
    f"{pkg}.algebra": ["Compose", "Sum", "Scaled", "Power"],
    f"{pkg}.identity": ["Identity"],
    f"{pkg}.shift": ["Shift"],
    f"{pkg}.fixed_point": ["FixedPoint"],
    f"{pkg}.norms": [
        "relative_l2", "relative_imbalance", "assert_idempotent"],
    f"{pkg}.propagator": ["Propagator"],
    f"{pkg}.time_average": ["TimeAverage"],
    f"{pkg}.adiabatic_ramping": ["AdiabaticRamping"],
    f"{pkg}.adiabatic_projection": ["AdiabaticProjection"],
    f"{pkg}.optimal_balance": ["OptimalBalance"],
    f"{pkg}.projection": [
        "EigenFunction", "EigenProjection", "ProjectionFactory"],
    f"{pkg}.balance_expansion": ["BalanceExpansion"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
