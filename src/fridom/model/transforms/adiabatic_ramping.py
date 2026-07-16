"""
AdiabaticRamping: deform a model between two operator configurations.

Description
-----------
The Tier-2 base surface for generalized adiabatic ramping
(``design/specs/model/08_state_transforms.md`` §10.9, decisions
AR-D1..D9). A *deformation* is a pair of endpoint parameter
assignments on one assembly — a *reference* configuration at
``lambda = 0`` and a *target* configuration at ``lambda = 1`` — and a
leg drives ``lambda(t) = rho((t - t0)/tau)`` by feeding Ramp-valued
parameter updates to an internal :class:`Propagator`. The transform
algebra stays clock-free (AR-D3): the ramp lives entirely in the
internal variant's stage-time parameters, never in the transform
surface.

The constructor always describes the **up** leg (reference -> target,
``dt > 0``). Two orthogonal accessors derive the other three legs,
each returning a **new** transform with a fresh internal variant:
``.down`` swaps the endpoints (lambda path ``1 -> 0``, ``dt > 0``) and
``.backward`` retraces the same lambda path with ``dt < 0``
(``fr.params.TIME_STEP`` sign flip + :meth:`Ramp.reversed`). They
compose (``ramp.down.backward``). ``.replace(**overrides)`` is the
frozen-config copy-with (OB's backward leg needs a different
``term_filter``).

``OptimalBalance`` is re-homed **onto** this surface (composition, not
subclass); this is the leg machinery it owns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.model.closures.base import ClosureBase
from fridom.model.errors import IrreversibleTermError
from fridom.model.time_dependent import Ramp, TimeDependent
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.propagator import Propagator

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.model.transforms.info import TransformCost, TransformInfo
    from fridom.model.transforms.signature import StateSignature
    from fridom.spatial.fields.vector_field import VectorField

# the AR-D6 irreversibility predicate: dissipative closures and any
# implicitly treated term are ill-posed integrated backward in time
_IRREVERSIBLE = terms.owned_by(ClosureBase) | terms.implicit


class AdiabaticRamping(StateTransform):

    """Deform a model between two operator configurations (§10.9)."""

    def __init__(
        self,
        model: object,
        *,
        ramps: Mapping[str, object],
        ramp_period: float,
        curve: str | Callable = "exp",
        steps: int | None = None,
        term_filter: Callable | None = None,
        updates: Mapping[str, object] | None = None,
        name: str | None = None,
        _backward: bool = False,
        _resolved: Mapping[str, TimeDependent] | None = None,
        _down: bool = False,
    ) -> None:
        """
        Build the up leg's internal ramped variant.

        Description
        -----------
        Tuple values in ``ramps`` are sugar for
        ``Ramp(v_ref, v_target, period=ramp_period, curve=curve)``; an
        explicit :class:`TimeDependent` is taken verbatim — its own
        ``t0``/``period`` window is the AR-D5 *interleaved* protocol
        form. The resolved Ramp-valued updates are merged with the
        passthrough ``updates`` and fed to an internal
        :class:`Propagator` over ``model.variant`` (never storing the
        passed model — §10.3 law 3). ``_backward``/``_resolved``/
        ``_down`` are private slots the leg accessors use; user code
        never sets them.

        Parameters
        ----------
        model : Model
            The assembly spec; a fresh internal variant is built from
            it (never stored/mutated — §10.3 law 3).
        ramps : Mapping[str, tuple | TimeDependent]
            ``{param_key: (v_ref, v_target) | TimeDependent}``; tuples
            become the up-leg Ramp, TimeDependents pass verbatim.
        ramp_period : float
            The ramping duration, in seconds; the leg advances
            ``steps`` internal steps and the tuple-sugar Ramps span
            ``[0, ramp_period]``.
        curve : str | Callable, optional
            The tuple-sugar ramp curve ("exp" | "cosine" | "linear")
            or a callable with ``shape(0)=0, shape(1)=1``
            (default: "exp").
        steps : int | None, optional
            Overrides the default ``max(1, round(ramp_period / |dt|))``
            internal step count (default: None).
        term_filter : Callable | None, optional
            A term predicate (``fr.terms``) threaded to the internal
            variant (default: None).
        updates : Mapping[str, object] | None, optional
            Passthrough (non-ramped) assembly-time parameter updates,
            merged under the ramp updates (default: None).
        name : str | None, optional
            The internal variant's report/log name prefix
            (default: ``"AdiabaticRamping"``).
        """
        dt = abs(float(model.parameters[params.TIME_STEP]))
        self._ramp_period = float(ramp_period)
        self._curve = curve
        self._term_filter = term_filter
        self._passthrough = dict(updates or {})
        self._name = name
        self._backward = bool(_backward)
        self._down = bool(_down)
        self._steps = (steps if steps is not None
                       else max(1, round(self._ramp_period / dt)))
        if _resolved is not None:
            self._resolved = dict(_resolved)
        else:
            self._resolved = self._resolve_ramps(ramps)
        merged = {**self._passthrough, **self._resolved}
        prefix = name or "AdiabaticRamping"
        suffix = ("/down" if self._down else "") + (
            "/backward" if self._backward else "/up")
        self._propagator = Propagator(
            model, steps=self._steps, backward=self._backward,
            updates=merged, term_filter=term_filter,
            name=f"{prefix}{suffix}")
        if self._backward:
            self._check_reversible()

    def _resolve_ramps(
        self, ramps: Mapping[str, object],
    ) -> dict[str, TimeDependent]:
        """Resolve the ``ramps`` dict to per-parameter Ramp curves."""
        resolved: dict[str, TimeDependent] = {}
        for key, spec in dict(ramps).items():
            if isinstance(spec, TimeDependent):
                resolved[key] = spec
                continue
            try:
                v_ref, v_target = spec
            except (TypeError, ValueError):
                raise TypeError(
                    f"ramps[{key!r}] must be a (v_ref, v_target) pair "
                    f"or a TimeDependent (fr.Ramp); got {spec!r}"
                ) from None
            resolved[key] = Ramp(
                v_ref, v_target, period=self._ramp_period,
                curve=self._curve)
        return resolved

    def _check_reversible(self) -> None:
        """Raise the AR-D6 taught error on a backward irreversible leg."""
        offending = self._propagator.model._resolve_tendency_filter(  # noqa: SLF001 — same-package term enumeration
            _IRREVERSIBLE)
        if offending:
            raise IrreversibleTermError(tuple(sorted(offending)))

    # ================================================================
    #  Leg accessors (each a NEW transform, fresh internal variant)
    # ================================================================
    @property
    def down(self) -> AdiabaticRamping:
        """
        The endpoint-swapped leg (lambda ``1 -> 0``, ``dt > 0``).

        Description
        -----------
        Maps target-side -> ref-side, forward in time. Each ramp's
        value endpoints swap and its window reflects within
        ``[0, ramp_period]`` (``t0 -> ramp_period - t0 - period``) —
        for the default symmetric curves this equals the time-reversed
        lambda profile played forward, so staggered windows re-order
        correctly (the AR-D5 composition-equivalence).

        Returns
        -------
        AdiabaticRamping
            The down leg.
        """
        reflected = {k: self._reflect(r)
                     for k, r in self._resolved.items()}
        return self._rebuild(reflected, down=not self._down)

    @property
    def backward(self) -> AdiabaticRamping:
        """
        The retraced leg (same lambda path, ``dt < 0``).

        Description
        -----------
        Flips ``fr.params.TIME_STEP`` on the internal variant and
        reflects every Ramp window across ``t = 0``
        (:meth:`Ramp.reversed`, spanning ``[-ramp_period, 0]``), so the
        backward leg retraces this leg's values exactly. Constructing
        it fires the AR-D6 guard (:class:`IrreversibleTermError`).

        Returns
        -------
        AdiabaticRamping
            The retraced leg.
        """
        reversed_ = {k: self._reverse(r)
                     for k, r in self._resolved.items()}
        return self._rebuild(reversed_, backward=not self._backward)

    def replace(self, **overrides: object) -> AdiabaticRamping:
        """
        Frozen-config copy-with: override any constructor keyword.

        Description
        -----------
        Rebuilds this leg with the given constructor keywords replaced
        (OB's backward leg needs a different ``term_filter``:
        ``forward.replace(term_filter=backward_filter).backward``).
        The current leg's resolved ramps, ``dt`` sign and step count
        are preserved unless overridden; passing ``ramps=`` re-resolves
        from scratch (a fresh up-shaped resolution).

        Parameters
        ----------
        **overrides : object
            Constructor keywords to replace (``ramps``, ``ramp_period``,
            ``curve``, ``steps``, ``term_filter``, ``updates``,
            ``name``).

        Returns
        -------
        AdiabaticRamping
            The reconfigured leg.
        """
        resolved = None if "ramps" in overrides else self._resolved
        return AdiabaticRamping(
            self._propagator.model,
            ramps=overrides.get("ramps", {}),
            ramp_period=overrides.get("ramp_period", self._ramp_period),
            curve=overrides.get("curve", self._curve),
            steps=overrides.get("steps", self._steps),
            term_filter=overrides.get("term_filter", self._term_filter),
            updates=overrides.get("updates", self._passthrough),
            name=overrides.get("name", self._name),
            _backward=self._backward, _resolved=resolved,
            _down=self._down)

    def _rebuild(
        self,
        resolved: Mapping[str, TimeDependent],
        *,
        backward: bool | None = None,
        down: bool | None = None,
    ) -> AdiabaticRamping:
        """Build a derived leg from this leg's own internal variant."""
        return AdiabaticRamping(
            self._propagator.model, ramps={},
            ramp_period=self._ramp_period, curve=self._curve,
            steps=self._steps, term_filter=self._term_filter,
            updates=self._passthrough, name=self._name,
            _backward=self._backward if backward is None else backward,
            _resolved=resolved,
            _down=self._down if down is None else down)

    def _reflect(self, ramp: TimeDependent) -> Ramp:
        """Endpoint-swap + window-reflect a Ramp within [0, period]."""
        r = self._as_ramp(ramp, "down")
        return Ramp(
            r.v1, r.v0, period=r.period,
            t0=self._ramp_period - r.t0 - r.period,
            curve=r._curve_spec)  # noqa: SLF001 — Ramp carries no public curve

    def _reverse(self, ramp: TimeDependent) -> Ramp:
        """Reflect a Ramp window across ``t = 0`` (retrace)."""
        return self._as_ramp(ramp, "backward").reversed()

    @staticmethod
    def _as_ramp(ramp: TimeDependent, leg: str) -> Ramp:
        """Require a concrete ``Ramp`` for a derived-leg reflection."""
        if not isinstance(ramp, Ramp):
            raise TypeError(
                f".{leg} reflects fr.Ramp windows, but a ramped "
                f"parameter is {ramp!r}; declare the ramp with the "
                "tuple sugar (v_ref, v_target) or an explicit fr.Ramp "
                "so it can be reflected")
        return ramp

    # ================================================================
    #  Declared structure (Tier 2, endo) — delegated to the Propagator
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: runs a model internally (never under a trace)."""
        return False

    @property
    def domain(self) -> StateSignature:
        """The internal model's PROGNOSTIC endo signature."""
        return self._propagator.domain

    @property
    def codomain(self) -> StateSignature:
        """The internal model's PROGNOSTIC endo signature."""
        return self._propagator.codomain

    @property
    def steps(self) -> int:
        """The number of internal model steps this leg runs."""
        return self._steps

    @property
    def is_backward(self) -> bool:
        """Whether the internal variant runs with a flipped dt."""
        return self._backward

    @property
    def is_down(self) -> bool:
        """Whether this is an endpoint-swapped (down) leg."""
        return self._down

    @property
    def model(self) -> object:
        """The owned internal variant (never the passed model)."""
        return self._propagator.model

    @property
    def propagator(self) -> Propagator:
        """The internal Propagator carrying the ramped variant."""
        return self._propagator

    @property
    def ramps(self) -> dict[str, TimeDependent]:
        """The resolved per-parameter ramp curves of this leg."""
        return dict(self._resolved)

    # ================================================================
    #  Application (delegated Tier-2 call, §10.3 law 1)
    # ================================================================
    def _evaluate(
        self, state: object,
    ) -> tuple[VectorField, TransformInfo]:
        """Run the internal ramped Propagator (reset; advance; read)."""
        return self._propagator._evaluate(state)  # noqa: SLF001 — delegated leaf call

    def cost(self) -> TransformCost:
        """Return the internal step count (the cost-opacity answer)."""
        return self._propagator.cost()

    def __repr__(self) -> str:
        """``AdiabaticRamping(params=..., ramp_steps=N, ...)``."""
        return (f"AdiabaticRamping(params={tuple(self._resolved)}, "
                f"ramp_steps={self._steps}, down={self._down}, "
                f"backward={self._backward})")
