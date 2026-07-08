"""
Propagator: run a model as a ``State -> State`` map (wave 7 B).

Description
-----------
The Tier-2 forward/backward building block (08 §10.5): wraps
``model.advance`` (and its ``TIME_STEP``-flipped reverse) as an endo
transform. Normative call semantics (§10.3 law 1):
``m.reset(); m.set_state(state); m.advance(N); return
m.state.prognostic`` — PROGNOSTIC-only read-back, the final internal
clock reported through *info*, never in the returned State. Isolation
by construction (§10.3 law 3): the constructor treats the passed model
as an assembly spec and builds its **own** internal variant (``io=()``
by construction), so two transforms never share a model; a Ramp-valued
``updates=`` slot (OB's ramped legs) and a ``backward`` dt sign flip
are assembly-time variant updates. Tier 2: a host object with the
inherited trace guard (``TraceError`` on tracer input).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.model import params
from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.info import TransformCost, TransformInfo
from fridom.framework2.transforms.signature import StateSignature

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

# the sign-agnostic runlen -> steps snap tolerance (matches
# Model._plan_run_steps: steps = ceil(k - eps))
_TARGET_EPS = 1e-9


class Propagator(StateTransform):

    """Run a model ``N`` steps as an endo ``State -> State`` map."""

    def __init__(
        self,
        model: object,
        *,
        steps: int | None = None,
        runlen: float | None = None,
        backward: bool = False,
        updates: Mapping[str, object] | None = None,
        term_filter: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """
        Build the internal variant and resolve the step count.

        Parameters
        ----------
        model : Model
            The assembly spec; a fresh internal variant is built from
            it (never stored/mutated — §10.3 law 3).
        steps : int | None, optional
            The number of internal steps; exactly one of
            ``steps``/``runlen`` (default: None).
        runlen : float | None, optional
            An (unsigned) model-time duration, snapped to
            ``ceil(runlen / |dt|)`` steps (default: None).
        backward : bool, optional
            Flip ``fr.params.TIME_STEP`` sign on the internal variant
            (the reverse leg) (default: False).
        updates : Mapping[str, object] | None, optional
            Assembly-time variant parameter updates, e.g. a Ramp-valued
            ``scaling.rossby`` for OB's ramped legs (default: None).
        term_filter : Callable | None, optional
            A term predicate threaded to the internal variant, e.g.
            ``fr.terms.linear`` (default: None).
        name : str | None, optional
            The internal variant's report/log name
            (default: ``"Propagator/internal"``).
        """
        dt = abs(float(model.parameters[params.TIME_STEP]))
        self._steps = self._resolve_steps(steps, runlen, dt)
        self._backward = bool(backward)
        merged = dict(updates or {})
        merged[params.TIME_STEP] = -dt if backward else dt
        self._model = model.variant(
            term_filter=term_filter, updates=merged,
            name=name or "Propagator/internal")
        self._prognostic = self._model.field_table.prognostic
        self._signature = StateSignature.of_prognostic(self._model)

    @staticmethod
    def _resolve_steps(
        steps: int | None, runlen: float | None, dt: float,
    ) -> int:
        """Reduce the ``steps``/``runlen`` target to a step count."""
        if (steps is None) == (runlen is None):
            raise ValueError(
                "Propagator takes exactly one of steps=/runlen= "
                f"(got steps={steps!r}, runlen={runlen!r})")
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) \
                    or steps < 0:
                raise ValueError(
                    f"steps must be a non-negative int, got {steps!r}")
            return steps
        if runlen <= 0.0:
            raise ValueError(
                f"runlen must be a positive duration, got {runlen!r}")
        return max(1, math.ceil(runlen / dt - _TARGET_EPS))

    # ================================================================
    #  Declared structure (Tier 2, endo)
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: runs a model internally (never under a trace)."""
        return False

    @property
    def domain(self) -> StateSignature:
        """The internal model's PROGNOSTIC endo signature."""
        return self._signature

    @property
    def codomain(self) -> StateSignature:
        """The internal model's PROGNOSTIC endo signature."""
        return self._signature

    @property
    def steps(self) -> int:
        """The number of internal model steps this Propagator runs."""
        return self._steps

    @property
    def is_backward(self) -> bool:
        """Whether the internal variant runs with a flipped dt."""
        return self._backward

    @property
    def model(self) -> object:
        """The owned internal variant (never the passed model)."""
        return self._model

    # ================================================================
    #  Application (the normative Tier-2 call, §10.3 law 1)
    # ================================================================
    def _evaluate(
        self, state: object,
    ) -> tuple[VectorField, TransformInfo]:
        """``reset(); set_state(state); advance(N)``; read PROGNOSTIC."""
        model = self._model
        model.reset()
        model.set_state(state)
        model.advance(self._steps)
        full = model.state
        result = VectorField(
            {name: full[name] for name in self._prognostic})
        info = TransformInfo(
            model_steps=self._steps,
            elapsed_model_time=float(model.clock.time),
            extra={"backward": self._backward})
        return result, info

    def cost(self) -> TransformCost:
        """Return the internal step count (the cost-opacity answer)."""
        return TransformCost(model_steps=self._steps)

    def __repr__(self) -> str:
        """``Propagator(steps=N, backward=...)``."""
        return (f"Propagator(steps={self._steps}, "
                f"backward={self._backward})")
