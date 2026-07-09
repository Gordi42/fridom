r"""
Shallow-water eigenmode projections (``sw.transforms``, wave 7 C).

Description
-----------
The Tier-1 vortical / wave / divergence projections as composable
``fr.StateTransform``s, wrapping the discrete eigenmode projector
(``sw.eigenmodes.Eigenmodes.projector``):

- ``VorticalProjection`` — the geostrophic mode ``P(0)``;
- ``WaveProjection`` — the inertia-gravity modes, built as the algebra
  sum ``P(+1) + P(-1)``;
- ``DivergenceProjection`` — the residual,
  ``(VorticalProjection + WaveProjection).complement``.

Each is a :class:`~fridom.framework2.transforms.projection.ProjectionFactory`
with **dual sources** — ``VorticalProjection(em)`` from an explicit
``Eigenmodes``, or ``VorticalProjection.from_model(model, at_time=...)``.

Staggered coefficient basis. The shallow-water modes are the
*discrete* C-grid eigenvectors, so the projection is a clean
round-trip on the model's physical **staggered** state (``u`` on the
east face, ``v`` on the north face, ``p`` on the centre): the
eigenmode kit's per-component transforms carry each component to its
own coefficient basis, the diagonal per-mode projector applies
there, and the backward transforms return to the nodal spaces. The
three modes span the three components at every wavenumber (the
``k = 0`` mean via the inertial patch), except the
interpolation-Nyquist planes where the geostrophic column is a
structural zero — ``DivergenceProjection`` picks up exactly that
Nyquist-vortical residual and is the zero map on Nyquist-free
states.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework2 as fr
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
    Project a staggered physical state onto the span of ``modes``.

    Description
    -----------
    Forward-transforms each component to its own staggered
    coefficient basis (the eigenmode kit's per-component
    transforms), applies ``sum_s em.projector(s)`` there (diagonal
    per wavenumber), and inverse-transforms back to each component's
    nodal space (layout and metadata kept).
    """
    kit = em._kit  # noqa: SLF001 — package-internal kit access
    coeff = State({
        name: kit.forward(name)(state[name]) for name in _COMPONENTS})
    projected = None
    for s in modes:
        contribution = em.projector(s)(coeff)
        projected = (contribution if projected is None
                     else projected + contribution)
    return State({
        name: state[name].with_data(
            kit.backward(name)(projected[name]).data)
        for name in _COMPONENTS})


def _signature(em: Eigenmodes) -> StateSignature:
    """Return the staggered endo signature ``(u, v, p)``."""
    grid = em.grid
    x, y = grid.names
    spaces = {"u": fr.Staggered(x), "v": fr.Staggered(y),
              "p": fr.Collocated()}
    components = tuple(
        (name, spaces[name].resolve(grid).bare)
        for name in _COMPONENTS)
    return StateSignature(grid=grid, components=components)


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
