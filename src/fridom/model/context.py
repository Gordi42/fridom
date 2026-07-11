"""
The per-substage step context.

Description
-----------
``StepContext``: the frozen context bundle every in-trace hook
receives — kwargs at the notebook boundary, ctx inside the trace.
Owning class spec: ``design/specs/model/classes/module.md``;
design source ``design/specs/model/03_time_stepping.md``
section 5.5.
"""
# Wave 2 C: StepContext
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Final

import jax.numpy as jnp

from fridom.framework.utils import jaxify

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.model.clock import Clock
    from fridom.model.terms import Treatment

# the fixed field set (also the pytree leaf order); StepContext is
# final and frozen — every attribute is set exactly once
_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "params", "clock", "dt", "stage_dt", "tendency_sums")


# ================================================================
#  StepContext
# ================================================================
@partial(jaxify, dynamic=_CONTEXT_FIELDS)
class StepContext:

    """
    Frozen per-substage context for in-trace hooks.

    Description
    -----------
    Built at P0 of every substage by the composed step — never by
    user code (pure diagnostics keep explicit kwargs; the binding
    layer converts). Concrete, final, and frozen: no methods beyond
    trivial accessors, no mutation, no back-references to the Model
    or modules (no coupling/run-loop state outside the carry).

    A jaxified pytree with all-scalar leaves in the base form, so
    the halo tracer needs zero mimicry machinery (it wraps *state*;
    ctx is scalars) and contexts differing only in dynamic scalar
    values share one trace. ``tendency_sums`` is the post-TENDENCY
    extension seam: populated (with the State-valued per-treatment
    sums) for ADVANCE/CONSTRAINT/DIAGNOSTIC stages by the composer
    (wave 3+); ``None`` for terms, SELF_UPDATE, and DIAGNOSE.

    Attributes
    ----------
    params : Mapping[str, jax.Array]
        ``eval_params(modules, stage_time)`` — live leaves read
        fresh at each stage time through the D2 binding table; Ramp
        values evaluated at RK sub-stage times.
    clock : Clock
        Traced clock; ``clock.time`` equals the stage time. Values
        are at the global width (02_rules, Clock precision): float64
        under the default x64-on run; in a float32 run the traced
        elapsed is re-anchored from the authoritative host-side
        float64 clock at every chunk boundary, and the exact integer
        ``it`` remains the primary key for triggers/schedules.
    dt : jax.Array
        Full step size — the cfl-type consumers' read surface.
    stage_dt : jax.Array
        Increment of the current advance (the ``p = phi / stage_dt``
        pressure normalization; backward runs thread sign
        conventions through it). NEVER ``dt_gamma``: the implicit
        ``solve`` receives gamma*dt as a separate stepper-supplied
        positional (gamma-specific: CN dt/2, SBDF2 2dt/3), never
        read from ctx.
    tendency_sums : Mapping[Treatment, Any] | None
        Per-treatment summed tendency contributions (State-valued)
        — populated for post-TENDENCY hooks only; ``None``
        otherwise (default: None).
    """

    params: Mapping[str, jax.Array]
    clock: Clock
    dt: jax.Array
    stage_dt: jax.Array
    tendency_sums: Mapping[Treatment, Any] | None

    def __init__(
        self,
        params: Mapping[str, jax.Array],
        clock: Clock,
        dt: Any,
        stage_dt: Any,
        tendency_sums: Mapping[Treatment, Any] | None = None,
    ) -> None:
        # a mutable dict is copied for isolation from later source
        # mutation; a frozen Params mapping (the in-trace path) is
        # kept as-is so its hinted MissingParameterError __getitem__
        # survives — copying to a plain dict would drop the hint
        self.params = dict(params) if isinstance(params, dict) else params
        self.clock = clock
        self.dt = jnp.asarray(dt)
        self.stage_dt = jnp.asarray(stage_dt)
        self.tendency_sums = tendency_sums

    def __setattr__(self, name: str, value: object) -> None:
        """Set a context field exactly once (frozen thereafter)."""
        # write-once over the fixed field set: __init__ and pytree
        # unflattening set fresh attributes; everything else raises
        if name in _CONTEXT_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"StepContext is frozen: cannot set {name!r} (contexts "
            "are built per substage at P0 by the composed step)")

    def __delattr__(self, name: str) -> None:
        """Raise: StepContext is frozen."""
        raise AttributeError(
            f"StepContext is frozen: cannot delete {name!r}")

    def __repr__(self) -> str:
        """Return a compact scalar-value repr for host-side logs."""
        sums = ("populated" if self.tendency_sums is not None
                else "None")
        return (f"StepContext(params={sorted(self.params)!r}, "
                f"clock={self.clock!r}, dt={self.dt!r}, "
                f"stage_dt={self.stage_dt!r}, "
                f"tendency_sums={sums})")
