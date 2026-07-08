r"""
Shallow-water eigenmode projections (``sw.transforms``, wave 7 C).

Description
-----------
The Tier-1 vortical / wave / divergence projections as composable
``fr.StateTransform``s, wrapping the collocated eigenmode projector
(``sw.eigenmodes.Eigenmodes.projector``):

- ``VorticalProjection`` — the geostrophic mode ``P(0)``;
- ``WaveProjection`` — the inertia-gravity modes, built as the algebra
  sum ``P(+1) + P(-1)``;
- ``DivergenceProjection`` — the residual,
  ``(VorticalProjection + WaveProjection).complement``.

Each is a :class:`~fridom.framework2.transforms.projection.ProjectionFactory`
with **dual sources** — ``VorticalProjection(em)`` from an explicit
``Eigenmodes``, or ``VorticalProjection.from_model(model, at_time=...)``.

Coefficient basis (wave-6 boundary). The shallow-water modes are the
*continuous* eigenvectors on a **collocated** coefficient basis, so the
projection is a clean round-trip on a physical *collocated* state
``(u, v, p on cell centres)``: forward-transform each component to the
coefficient basis, apply the diagonal per-mode projector, inverse-
transform back. Wrapping the model's *staggered* physical state needs
the deferred discrete staggered spectral transforms (flagged in
``sw.eigenmodes``), so this iteration validates on the collocated state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.transforms.projection import (
    EigenProjection,
    ProjectionFactory,
)
from fridom.framework2.transforms.signature import StateSignature
from fridom.shallowwater2.eigenmodes import from_model
from fridom.shallowwater2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.transforms.base import StateTransform
    from fridom.shallowwater2.eigenmodes import Eigenmodes

#: the shallow-water prognostic components, in State order.
_COMPONENTS = ("u", "v", "p")


# ================================================================
#  The coefficient-basis projector (forward -> project -> backward)
# ================================================================
def _project(
    em: Eigenmodes, modes: tuple[int, ...], state: State,
) -> State:
    r"""
    Project a collocated physical state onto the span of ``modes``.

    Description
    -----------
    Forward-transforms each component to the collocated coefficient
    basis, applies ``sum_s em.projector(s)`` there (diagonal per
    wavenumber), and inverse-transforms back to the cell-centre space.
    """
    transform = em.transform
    coeff = State({
        name: transform.forward(state[name]) for name in _COMPONENTS})
    projected = None
    for s in modes:
        contribution = em.projector(s)(coeff)
        projected = (contribution if projected is None
                     else projected + contribution)
    return State({
        name: transform.backward(projected[name])
        for name in _COMPONENTS})


def _signature(em: Eigenmodes) -> StateSignature:
    """Return the collocated (cell-centre) endo signature ``(u, v, p)``."""
    center = em.center_space.bare
    components = tuple((name, center) for name in _COMPONENTS)
    return StateSignature(grid=em.grid, components=components)


# ================================================================
#  Single-mode and composite builders
# ================================================================
def mode_projection(
    em: Eigenmodes, s: int, *, name: str | None = None,
) -> EigenProjection:
    """Return the single-mode projection ``P(s)`` as a transform."""
    return EigenProjection(
        eigenmodes=em,
        modes=(s,),
        signature=_signature(em),
        project_fn=_project,
        name=name or f"P({s:+d})")


def _build_vortical(em: Eigenmodes) -> StateTransform:
    """Return the geostrophic projection ``P(0)``."""
    return mode_projection(em, 0, name="VorticalProjection")


def _build_wave(em: Eigenmodes) -> StateTransform:
    """Return the inertia-gravity projection ``P(+1) + P(-1)`` (merged)."""
    return mode_projection(em, 1) + mode_projection(em, -1)


def _build_divergence(em: Eigenmodes) -> StateTransform:
    """Return the residual ``(P_vortical + P_wave).complement``."""
    return (_build_vortical(em) + _build_wave(em)).complement


# ================================================================
#  Public dual-source projections
# ================================================================
VorticalProjection = ProjectionFactory(
    _build_vortical, from_model, "VorticalProjection")
WaveProjection = ProjectionFactory(
    _build_wave, from_model, "WaveProjection")
DivergenceProjection = ProjectionFactory(
    _build_divergence, from_model, "DivergenceProjection")
