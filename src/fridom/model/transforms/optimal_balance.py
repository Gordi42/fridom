r"""
OptimalBalance: nonlinear balancing by ramped propagation (wave 7 B).

Description
-----------
The Tier-2 preset that projects onto the (slow) balanced manifold via
the optimal-balance method (08 §10.5). Composed **on**
:class:`AdiabaticRamping` (AR-D4, §10.9 "Reconciliation with §10.5"):
the forward (up) leg is ``AdiabaticRamping(model, envelope=True,
...)`` — it ramps the **nonlinearity as a whole** (a term-envelope
:math:`\rho(t)` on the ``~fr.terms.linear & fr.terms.explicit``
terms, §C), so OB works identically on dimensional models (no
scaling parameter bound) and nondimensional ones. A bound scaling
parameter :math:`\varepsilon` (constant *or* ``fr.Ramp``-valued) is
never touched: the effective nonlinearity mid-ramp is
:math:`\rho(t)\,\varepsilon`. The backward leg is
``forward.replace(term_filter=backward_filter).backward`` — its
retrace with a flipped ``TIME_STEP`` and reversed ramp window. The
legs' internal Propagators form the
``ramp_cycle = forward @ base @ backward`` (the algebra at work). The
iteration is a :class:`FixedPoint` **factory** whose per-iterate
transform is the base-point exchange
``Shift(z_base) @ (Identity - base) @ ramp_cycle``; the factory form
absorbs ``update_base_point`` cleanly (§10.2).

The injected ``base_projection`` is any endo :class:`StateTransform`
(a Tier-1 eigenmode projection at cutover; an ``Identity`` / test
double for the machinery), so OB does not depend on the wave-7 C
projections. Old piecewise-constant ramping becomes continuous
stage-time ``Ramp`` evaluation. Tier 2: a host object with the
inherited trace guard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fridom.model.transforms.adiabatic_ramping import AdiabaticRamping
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.fixed_point import FixedPoint
from fridom.model.transforms.identity import Identity
from fridom.model.transforms.info import TransformCost, TransformInfo
from fridom.model.transforms.norms import relative_l2
from fridom.model.transforms.shift import Shift

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.transforms.propagator import Propagator
    from fridom.spatial.fields.vector_field import VectorField


class OptimalBalance(StateTransform):

    """Balance a state onto the slow manifold by ramped propagation."""

    def __init__(
        self,
        model: object,
        base_projection: StateTransform,
        *,
        ramp_period: float,
        ramp: str = "exp",
        max_it: int = 3,
        tol: float = 1e-9,
        update_base_point: bool = True,
        norm: Callable[[VectorField, VectorField], float] = relative_l2,
        on_divergence: Literal[
            "stop_best", "raise", "ignore"] = "stop_best",
        filter: Callable | None = None,  # noqa: A002 — spec-fixed keyword
        backward_filter: Callable | None = None,
        name: str | None = None,
    ) -> None:
        """
        Build the two ramped legs and configure the iteration.

        Parameters
        ----------
        model : Model
            The assembly spec; the two ramped legs build their own
            internal variants from it (never stored/mutated).
        base_projection : StateTransform
            The injected endo projector onto the base coordinate; must
            be idempotent for the base-coordinate invariant to hold.
        ramp_period : float
            The ramping duration, in seconds; snapped to
            ``round(ramp_period / |dt|)`` internal steps per leg.
        ramp : str, optional
            The ramp curve ("exp" | "cosine" | "linear") applied to
            the term envelope ``ramping.envelope`` (default: "exp").
        max_it : int, optional
            The maximum fixed-point iterations (default: 3).
        tol : float, optional
            The convergence tolerance (default: 1e-9).
        update_base_point : bool, optional
            Recompute the base coordinate from the current iterate each
            iteration (OBTA); otherwise hold the initial one
            (default: True).
        norm : Callable, optional
            The fixed-point convergence norm (default:
            :func:`relative_l2`).
        on_divergence : {"stop_best", "raise", "ignore"}, optional
            The fixed-point divergence policy (default: "stop_best").
        filter : Callable | None, optional
            A term predicate threaded to the forward leg
            (default: None).
        backward_filter : Callable | None, optional
            A term predicate threaded to the backward leg; ``None``
            reuses ``filter`` (default: None).
        name : str | None, optional
            The internal legs' report/log name prefix
            (default: ``"OptimalBalance"``).
        """
        prefix = name or "OptimalBalance"
        # OB is composed ON AdiabaticRamping (AR-D4, §10.9): the up
        # leg owns the envelope build + step snapping — envelope=True
        # ramps the nonlinearity as a whole (rho(t) on the
        # ~linear & explicit terms, §C), never a scaling parameter,
        # so OB balances dimensional and (Ramp-)nondimensional models
        # alike. A purely linear model is refused at assembly (the
        # taught empty-match error) instead of silently running a
        # no-ramp cycle. The backward leg is the "replace-then-
        # backward" retrace (flipped TIME_STEP, reversed window).
        # backward_filter=None reuses the forward filter. The AR-D6
        # guard fires on the dt<0 leg — inert on OB's closure-free
        # legs; a closured model with no backward_filter raises (the
        # sanctioned §10.8 behavior).
        forward_leg = AdiabaticRamping(
            model, envelope=True, ramp_period=ramp_period,
            curve=ramp, term_filter=filter, name=prefix)
        bwd_filter = (filter if backward_filter is None
                      else backward_filter)
        backward_leg = forward_leg.replace(term_filter=bwd_filter).backward
        # expose the legs' internal Propagators so ob.forward/.backward
        # keep their pinned public type and the ramp cycle composes
        # Propagators exactly as before.
        self._forward = forward_leg.propagator
        self._backward = backward_leg.propagator
        self._ramp_steps = forward_leg.steps
        self._base = base_projection
        self._max_it = max_it
        self._tol = float(tol)
        self._update_base_point = bool(update_base_point)
        self._norm = norm
        self._on_divergence = on_divergence

    # ================================================================
    #  Declared structure (Tier 2, endo)
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: runs models internally (never under a trace)."""
        return False

    @property
    def domain(self) -> object:
        """The ramp cycle's PROGNOSTIC endo signature."""
        return self._backward.domain

    @property
    def codomain(self) -> object:
        """The ramp cycle's PROGNOSTIC endo signature."""
        return self._forward.codomain

    @property
    def base(self) -> StateTransform:
        """The injected base-coordinate projection."""
        return self._base

    @property
    def forward(self) -> Propagator:
        """The forward (linear -> nonlinear) ramped leg."""
        return self._forward

    @property
    def backward(self) -> Propagator:
        """The backward (nonlinear -> linear) ramped leg."""
        return self._backward

    @property
    def ramp_cycle(self) -> StateTransform:
        """``forward @ base @ backward`` — one balancing cycle."""
        return self._forward @ self._base @ self._backward

    # ================================================================
    #  Application (the fixed-point base-point exchange)
    # ================================================================
    def _exchange_factory(
        self, ramp_cycle: StateTransform, z_base_initial: VectorField,
    ) -> Callable[[VectorField], StateTransform]:
        """Build the per-iterate base-point-exchange transform factory."""
        base = self._base

        def factory(iterate: VectorField) -> StateTransform:
            """``Shift(z_base) @ (I - base) @ ramp_cycle`` per iterate."""
            z_base = (base(iterate) if self._update_base_point
                      else z_base_initial)
            return Shift(z_base) @ (Identity() - base) @ ramp_cycle

        return factory

    def _evaluate(
        self, state: object,
    ) -> tuple[VectorField, TransformInfo]:
        """Iterate the ramped base-point exchange to a fixed point."""
        ramp_cycle = self.ramp_cycle
        z_base_initial = self._base(state)
        factory = self._exchange_factory(ramp_cycle, z_base_initial)
        iteration = FixedPoint(
            factory, tol=self._tol, max_it=self._max_it,
            norm=self._norm, on_divergence=self._on_divergence)
        result, fp_info = iteration.call_with_info(state)
        # FixedPoint's own record does not roll its per-iteration
        # child steps up to its node, so aggregate the executed model
        # steps from the iteration children here (the cost-opacity
        # answer for the actually-run iterations).
        model_steps = sum(
            child.model_steps for _, child in fp_info.children)
        elapsed = sum(
            child.elapsed_model_time for _, child in fp_info.children)
        info = TransformInfo(
            iterations=fp_info.iterations, errors=fp_info.errors,
            model_steps=model_steps, elapsed_model_time=elapsed,
            extra=dict(fp_info.extra),
            children=(("fixed_point", fp_info),))
        return result, info

    def cost(self) -> TransformCost:
        """``max_it *`` the two legs' step cost, flagged upper-bound."""
        cycle = self._forward.cost() + self._backward.cost()
        total = cycle * self._max_it
        return TransformCost(
            model_steps=total.model_steps, upper_bound=True)

    def __repr__(self) -> str:
        """``OptimalBalance(base=..., ramp_steps=N, max_it=M)``."""
        return (f"OptimalBalance(base={self._base!r}, "
                f"ramp_steps={self._ramp_steps}, "
                f"max_it={self._max_it})")
