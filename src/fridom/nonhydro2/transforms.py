r"""
Nonhydrostatic eigenmode projections (``nh.transforms``, wave 7 C).

Description
-----------
The Tier-1 vortical / wave / divergence projections as composable
``fr.StateTransform``s, wrapping the discrete eigenmode projector
(``nh.eigenmodes.Eigenmodes.projector``):

- ``VorticalProjection`` — the geostrophic mode ``P(0)``;
- ``WaveProjection`` — the inertia-gravity modes, built as the algebra
  sum ``P(+1) + P(-1)``;
- ``DivergenceProjection`` — the residual,
  ``(VorticalProjection + WaveProjection).complement``. Unlike shallow
  water (a complete 3-mode basis), the four nonhydro components leave a
  genuine unbalanced residual, so this projection is **non-trivial**.

Each is a :class:`~fridom.framework2.transforms.projection.ProjectionFactory`
with dual sources — ``VorticalProjection(em)`` from an explicit
``Eigenmodes``, or ``VorticalProjection.from_model(model, at_time=...)``.

Staggered spectral basis. The nonhydro modes are the *discrete* C-grid
eigenvectors, and the projection is a clean per-component round-trip
through the grid's own transforms: each ``(u, v, w, b)`` component is
forward-transformed on its **own** physical space (``u``/``v``/``w``
face-staggered, ``b`` collocated) via the eigenmode kit
(``em.kit.forward(name)``), ``sum_s em.projector(s)`` acts on the
coefficient state (diagonal per wavenumber — the half-cell staggering
lives in the eigenvectors' operator symbols), and the backward
transforms return to the staggered physical spaces. The coefficient
layout is the grid's real-FFT layout — half spectrum on the first
transformed axis, full-complex later axes — so on a real state a
single-branch projection is Hermitian-closed by the real part of the
backward transform (see :func:`mode_projection`). The transforms carry
the concrete staggered endo ``StateSignature`` over ``(u, v, w, b)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.transforms.projection import (
    EigenProjection,
    ProjectionFactory,
)
from fridom.framework2.transforms.signature import StateSignature
from fridom.nonhydro2.eigenmodes import from_model
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.transforms.base import StateTransform
    from fridom.nonhydro2.eigenmodes import Eigenmodes

#: the nonhydro prognostic components the projector acts on.
_COMPONENTS = ("u", "v", "w", "b")


# ================================================================
#  The staggered projector (forward -> project -> backward)
# ================================================================
def _project(
    em: Eigenmodes, modes: tuple[int, ...], state: State,
) -> State:
    r"""
    Project a staggered physical state onto the span of ``modes``.

    Description
    -----------
    Forward-transforms each ``(u, v, w, b)`` component on its own
    staggered space through the eigenmode kit, applies
    ``sum_s em.projector(s)`` on the coefficient state (diagonal per
    wavenumber), inverse-transforms back, and takes the real part —
    the Hermitian closure of single-branch projections on the stored
    rfft half-lattice (see :func:`mode_projection`). On a walled
    grid the kit spaces carry the analysis parity tags while the
    state arrives on the model's (BC-free ``u``/``v``/``b``) spaces,
    so each component is retagged onto its kit space before the
    forward transform and back after the backward one — identity on
    periodic grids.
    """
    kit = em.kit
    forward = {name: kit.forward(name) for name in _COMPONENTS}
    coeff = State({
        name: forward[name](state[name].retag(forward[name].domain))
        for name in _COMPONENTS})
    projected = None
    for s in modes:
        contribution = em.projector(s)(coeff)
        projected = (contribution if projected is None
                     else projected + contribution)
    return State({
        name: kit.backward(name)(projected[name]).real
        .retag(state[name])
        for name in _COMPONENTS})


def _signature(em: Eigenmodes) -> StateSignature:
    """Return the staggered endo signature ``(u, v, w, b)``.

    Description
    -----------
    Each component on its own **model-facing** physical space —
    ``u``/``v``/``w`` face-staggered, ``b`` collocated, with ``w``'s
    Dirichlet wall tag on a walled grid (``em.physical_space``); the
    kit's parity-tagged analysis spaces stay internal to
    :func:`_project`'s retag round-trip.
    """
    components = tuple(
        (name, em.physical_space(name)) for name in _COMPONENTS)
    return StateSignature(grid=em.grid, components=components)


# ================================================================
#  Single-mode and composite builders
# ================================================================
def mode_projection(
    em: Eigenmodes, s: int, *, name: str | None = None,
) -> EigenProjection:
    r"""
    Return the single-mode projection ``P(s)`` as a transform.

    Description
    -----------
    Branch labelling: ``P(s)`` projects onto the eigenvector
    ``em.q(s)``, which pairs with the eigenvalue ``+i omega(s)`` of
    the linearized tendency (the operator-sourced convention; the
    previous hand-coded modes paired ``q(s)`` with ``-i omega(s)``,
    so the individual ``+1``/``-1`` branches are swapped relative to
    that labelling while ``P(0)`` and ``P(+1) + P(-1)`` are
    unchanged).

    Single-branch semantics on a real state: the state's spectrum is
    stored on the grid's rfft half-lattice, so ``P(s)`` applies the
    ``s``-branch projector on the stored half-lattice while the
    implicit conjugate half carries the mirrored ``-s`` branch; the
    real part of the backward transform realizes exactly this
    Hermitian closure, and the result is a **real** physical field
    (not the complex single-branch field of a full-complex lattice).
    ``P(0)`` and the merged ``P(+1) + P(-1)`` family are closed under
    the mirror pairing, hence unaffected.
    """
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
