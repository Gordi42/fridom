"""
AdiabaticProjection: project where no spectral decomposition exists.

Description
-----------
The Tier-2 preset of the paper's appendix B
(``design/specs/model/08_state_transforms.md`` §10.9, decision AR-D8):
to project onto the slow manifold of a system whose discretized
operator has no reference-frame spectral decomposition (a beta plane,
say), ramp the *linearized* system adiabatically to a reference
configuration where a projector **does** exist, project there, and
ramp back. The cycle is

.. code-block:: python

    P_adiab = leg @ reference_projection @ leg.backward

with ``leg`` the **up** leg (reference -> target, ``dt > 0``) and
``leg.backward`` its retrace (target -> reference, ``dt < 0``).

**Phase neutrality (AR-D8, normative).** The away-leg runs *backward*
in time (``leg.backward``) and the return-leg *forward* (``leg``), so
the mode phases a propagator leg advances cancel up to diabatic
leakage. A forward-forward cycle (``leg @ P_ref @ leg.down``) is a
*propagator*, not a projector: slow modes at the target end are not
stationary (equatorial Rossby modes have nonzero frequency), so it
returns the projected state advanced by ~``2 tau`` of linear dynamics
and ``P @ P`` drifts further from ``P``. Running the away-leg backward
cancels that phase exactly.

Consequences (all §10.9): the injected ``reference_projection`` and the
built ``leg`` keep the linearize-and-filter step visible in user code;
the constructor validates the leg's model with
:func:`~fridom.model.term_predicates.require_linear_operator` (an
incomplete ``L`` — e.g. the conserving route-B rotation — would make
the ramp integrate a different system than ``L`` describes) and builds
``leg.backward`` eagerly, so the AR-D6 irreversibility guard fires at
construction (a linearized model may still carry *linear* dissipation,
which must be filtered out of the legs). ``P_adiab`` is declared
``idempotent=True`` (a projector by contract) but is exact only up to
diabatic leakage — exponentially small in ``tau`` — so
:func:`~fridom.model.transforms.norms.assert_idempotent` and the
``P @ P`` lint use a **leg-dependent documented tolerance**. Cost: two
linear-model integrations per application, visible via
:meth:`cost`/``repr``. Tier 2: a host object with the inherited trace
guard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.term_predicates import require_linear_operator
from fridom.model.transforms.adiabatic_ramping import AdiabaticRamping
from fridom.model.transforms.base import StateTransform

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.transforms.info import TransformCost, TransformInfo
    from fridom.model.transforms.signature import StateSignature
    from fridom.spatial.fields.vector_field import VectorField


class AdiabaticProjection(StateTransform):

    """Adiabatic (appendix-B) slow projection (§10.9, AR-D8)."""

    def __init__(
        self,
        leg: AdiabaticRamping,
        reference_projection: StateTransform,
        *,
        name: str | None = None,
    ) -> None:
        """
        Wrap a built up leg + reference projector as a projection.

        Description
        -----------
        Validates the ``leg``'s linear operator, constructs the
        retraced away-leg ``leg.backward`` eagerly (firing the AR-D6
        guard at construction), and composes the phase-neutral cycle
        ``leg @ reference_projection @ leg.backward`` (whose eager
        signature checks pin the leg <-> projector compatibility).

        Parameters
        ----------
        leg : AdiabaticRamping
            The **up** leg (reference -> target, ``dt > 0``); a fresh
            ``leg.backward`` retraces it. Build it on the linearized
            model (``term_filter=fr.model.term_predicates.linear``, or a
            pre-linearized ``model.variant``) — the linearize step
            stays visible in user code.
        reference_projection : StateTransform
            The endo projector at the leg's **reference** end
            (``lambda = 0``); its signature must match the leg's
            PROGNOSTIC signature.
        name : str | None, optional
            The transform's report name (default:
            ``"AdiabaticProjection"``).

        Raises
        ------
        TypeError
            If ``leg`` is not an :class:`AdiabaticRamping`.
        LinearOperatorGapError
            If the leg's model carries linear physics outside ``L``
            (:func:`require_linear_operator`).
        IrreversibleTermError
            If ``leg.backward`` retains a dissipative/implicit term
            (the AR-D6 guard, fired eagerly here).
        SignatureMismatchError
            If the reference projector's signature does not match the
            leg's.
        """
        if not isinstance(leg, AdiabaticRamping):
            raise TypeError(
                "AdiabaticProjection takes a BUILT AdiabaticRamping up "
                f"leg (reference -> target), got {leg!r}; build the "
                "linear ramp leg first, e.g. AdiabaticRamping(model, "
                "ramps={...}, term_filter=fr.model.term_predicates."
                "linear)")
        if not isinstance(reference_projection, StateTransform):
            raise TypeError(
                "AdiabaticProjection needs a StateTransform reference "
                f"projector, got {reference_projection!r}")
        require_linear_operator(leg.model, consumer="AdiabaticProjection")
        self._leg = leg
        self._reference_projection = reference_projection
        self._name = name or "AdiabaticProjection"
        # eager (fires the AR-D6 guard at construction) — a linearized
        # model may still carry a LINEAR closure (e.g. diffusion).
        self._backward_leg = leg.backward
        # the phase-neutral cycle (AR-D8): away backward, project at the
        # reference end, return forward. The @ checks the leg <->
        # projector signatures eagerly (compose time).
        self._cycle = leg @ reference_projection @ self._backward_leg

    # ================================================================
    #  Declared structure (Tier 2, endo, idempotent by contract)
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: runs models internally (never under a trace)."""
        return False

    @property
    def idempotent(self) -> bool:
        """Declared: a projector by contract (exact up to leakage)."""
        return True

    @property
    def domain(self) -> StateSignature | None:
        """The cycle's PROGNOSTIC endo signature (the target end)."""
        return self._cycle.domain

    @property
    def codomain(self) -> StateSignature | None:
        """The cycle's PROGNOSTIC endo signature (the target end)."""
        return self._cycle.codomain

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def leg(self) -> AdiabaticRamping:
        """The up (reference -> target) leg."""
        return self._leg

    @property
    def backward_leg(self) -> AdiabaticRamping:
        """The retraced away-leg (``leg.backward``)."""
        return self._backward_leg

    @property
    def reference_projection(self) -> StateTransform:
        """The injected reference-end projector."""
        return self._reference_projection

    @property
    def cycle(self) -> StateTransform:
        """``leg @ reference_projection @ leg.backward``."""
        return self._cycle

    # ================================================================
    #  Application (delegated to the phase-neutral cycle, §10.3 law 1)
    # ================================================================
    def _evaluate(
        self, state: object,
    ) -> tuple[VectorField, TransformInfo]:
        """Run the cycle; its info mirrors the two legs + projector."""
        return self._cycle.call_with_info(state)

    def cost(self) -> TransformCost:
        """Two leg integrations per application (the projector is free)."""
        return self._cycle.cost()

    def __repr__(self) -> str:
        """``AdiabaticProjection(leg=..., P_ref=..., steps=N)``."""
        return (f"AdiabaticProjection(leg={self._leg!r}, "
                f"P_ref={self._reference_projection!r}, "
                f"steps={self._leg.steps})")
