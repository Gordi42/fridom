"""
TimeAverage: geostrophic projection by time-averaging (wave 7 B).

Description
-----------
The Tier-2 preset that removes oscillatory (inertia-gravity wave)
content, leaving the stationary (geostrophic) mode (08 §10.5). For a
constant Coriolis parameter the linear geostrophic mode is constant in
time while the wave modes oscillate, so a flat endpoint-inclusive mean
over an inertial period cancels the waves; nested passes with
descending periods stagger the sinc zeros for cleaner removal, and an
optional backward-forward pass symmetrizes.

The old algorithm ported faithfully (``fridom.framework.projection.
GeostrophicTimeAverage``), with the **signed parity delta pinned for
cutover**: the default ``filter=fr.terms.linear`` drops Smagorinsky
(nonlinear), which the old "linear" twin wrongly kept running. The
accumulation is the iteration-1 host ``advance(1)`` loop (old cost,
zero machinery). Tier 2: a host object with the inherited trace guard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.model.time_dependent import TimeDependent
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.info import TransformCost, TransformInfo
from fridom.model.transforms.signature import StateSignature
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class TimeAverage(StateTransform):

    """Project onto the geostrophic subspace by time-averaging."""

    def __init__(
        self,
        model: object,
        *,
        period: float | None = None,
        n_ave: int = 2,
        equidistant: bool = True,
        backward_forward: bool = False,
        filter: Callable | None = None,  # noqa: A002 — spec-fixed keyword
        name: str | None = None,
    ) -> None:
        """
        Build the filtered internal variant(s) and the period plan.

        Parameters
        ----------
        model : Model
            The assembly spec; a filtered internal variant is built
            from it (never stored/mutated — §10.3 law 3).
        period : float | None, optional
            The maximum averaging period, in seconds; ``None`` reads
            the inertial period ``2*pi / |coriolis.f0|`` (default:
            None).
        n_ave : int, optional
            The number of averaging passes (default: 2).
        equidistant : bool, optional
            Split the passes into descending periods spanning
            ``[period/2, period]``; otherwise all passes use ``period``
            (default: True).
        backward_forward : bool, optional
            Follow each forward pass with a ``TIME_STEP``-flipped
            backward pass (symmetrization) (default: False).
        filter : Callable | None, optional
            The term predicate for the internal variant; ``None`` reads
            ``fr.terms.linear`` (drops advection and closures — the
            Smagorinsky parity delta) (default: None).
        name : str | None, optional
            The internal variant's report/log name prefix
            (default: ``"TimeAverage"``).
        """
        if isinstance(n_ave, bool) or not isinstance(n_ave, int) \
                or n_ave < 1:
            raise ValueError(
                f"n_ave must be a positive int, got {n_ave!r}")
        term_filter = terms.linear if filter is None else filter
        dt = abs(float(model.parameters[params.TIME_STEP]))
        period = self._resolve_period(model, period)
        periods = self._plan_periods(period, n_ave, equidistant)
        self._n_steps = tuple(
            max(1, int(np.ceil(p / dt))) for p in periods)
        self._backward_forward = bool(backward_forward)
        prefix = name or "TimeAverage"
        self._forward = model.variant(
            term_filter=term_filter, name=f"{prefix}/forward")
        # the backward pass chains off the forward variant's own module
        # tree: assembly binds module instances in place, so a second
        # variant must build from the forward variant rather than twice
        # from the passed model.
        self._backward = (
            self._forward.variant(
                term_filter=term_filter,
                updates={params.TIME_STEP: -dt},
                name=f"{prefix}/backward")
            if backward_forward else None)
        self._prognostic = self._forward.field_table.prognostic
        self._signature = StateSignature.of_prognostic(self._forward)

    @staticmethod
    def _resolve_period(model: object, period: float | None) -> float:
        """Resolve ``period=None`` to the inertial period."""
        if period is not None:
            if period <= 0.0:
                raise ValueError(
                    f"period must be positive, got {period!r}")
            return float(period)
        parameters = model.parameters
        f0_name = params.CORIOLIS_F0
        if f0_name not in parameters:
            raise ValueError(
                "period=None reads the inertial period 2*pi/f0, but "
                f"this model provides no {f0_name!r} parameter; pass "
                "period=<seconds> explicitly")
        f0_value = parameters[f0_name]
        if isinstance(f0_value, TimeDependent):
            raise ValueError(  # noqa: TRY004 — a period=None config error, the sibling raises above are ValueError
                "period=None reads the inertial period 2*pi/f0, but "
                "coriolis.f0 is time-dependent (an fr.Ramp), so the "
                "inertial period is not a single constant; pass "
                "period=<seconds> explicitly")
        f0 = abs(float(f0_value))
        if f0 == 0.0:
            raise ValueError(
                "period=None reads the inertial period 2*pi/f0, but "
                "coriolis.f0 is zero; pass period=<seconds> explicitly")
        return float(2.0 * np.pi / f0)

    @staticmethod
    def _plan_periods(
        period: float, n_ave: int, equidistant: bool,
    ) -> np.ndarray:
        """Descending periods (staggered sinc zeros), or all ``period``."""
        if equidistant:
            return np.linspace(period / 2, period, n_ave + 1)[1:][::-1]
        return np.ones(n_ave) * period

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
    def n_steps(self) -> tuple[int, ...]:
        """The per-pass step counts (descending periods)."""
        return self._n_steps

    # ================================================================
    #  Application (the nested averaging passes)
    # ================================================================
    def _prognostic_of(self, state: object) -> VectorField:
        """Extract the PROGNOSTIC subset of a state as a VectorField."""
        return VectorField(
            {name: state[name] for name in self._prognostic})

    def _average(
        self, model: object, z: VectorField, n_its: int,
    ) -> VectorField:
        """Flat endpoint-inclusive mean over ``n_its`` steps."""
        model.reset()
        model.set_state(z)
        acc = z
        for _ in range(n_its):
            model.advance(1)
            acc = acc + self._prognostic_of(model.state)
        return acc / (n_its + 1)

    def _evaluate(
        self, state: object,
    ) -> tuple[VectorField, TransformInfo]:
        """Run the nested forward (and optional backward) passes."""
        z_ave = self._prognostic_of(state)
        model_steps = 0
        for n_its in self._n_steps:
            z_ave = self._average(self._forward, z_ave, n_its)
            model_steps += n_its
            if self._backward is not None:
                z_ave = self._average(self._backward, z_ave, n_its)
                model_steps += n_its
        info = TransformInfo(
            model_steps=model_steps,
            extra={"passes": len(self._n_steps),
                   "backward_forward": self._backward_forward})
        return z_ave, info

    def cost(self) -> TransformCost:
        """Return the total internal step count over every pass."""
        per_pass = 2 if self._backward is not None else 1
        return TransformCost(
            model_steps=per_pass * sum(self._n_steps))

    def __repr__(self) -> str:
        """``TimeAverage(n_steps=..., backward_forward=...)``."""
        return (f"TimeAverage(n_steps={self._n_steps}, "
                f"backward_forward={self._backward_forward})")
