r"""
Nonhydrostatic eigenmode projections (``nh.transforms``, wave 7 C).

Description
-----------
The Tier-1 vortical / wave / kelvin / divergence projections as
composable ``fr.StateTransform``s, with **dual eigenmode backends**
routed on the eigenmodes object type:

- the analytic operator-sourced ``nh.eigenmodes.Eigenmodes`` of the
  fully periodic / walled-vertical grid (per-mode spectral
  projector, the original path, bitwise unchanged);
- the labeled numeric ``nh.ChannelEigenmodes`` of the horizontally
  walled channel (the shared dense-column engine path of
  :mod:`fridom.model.eigenbasis`: partial-axis Fourier
  along the two periodic axes, per-plane column projection under
  the energy metric).

The public factories:

- ``VorticalProjection`` — the geostrophic mode ``P(0)``
  (analytic), or the labeled **vortical** family (channel; the
  divergence-complement zero modes are a separate ``constraint``
  family and are never selected);
- ``WaveProjection`` — the inertia-gravity modes ``P(+1) + P(-1)``
  (analytic), or the labeled **wave+ and wave-** branches only
  (channel; Kelvin is a separate family);
- ``KelvinProjection`` — the boundary-trapped **kelvin+ and
  kelvin-** pair (channel only; grids without horizontal walls
  raise: no walls, no Kelvin family);
- ``DivergenceProjection`` — the residual complement of every named
  physical family. The nonhydro components genuinely carry
  unbalanced content, so this projection is **non-trivial** on both
  backends: on the analytic tier the family spans exactly the
  Leray-constrained subspace minus the structurally excluded
  ``k_h = 0`` inertial strata and the ``k = 0`` mean (the even-grid
  Nyquist steady strata are covered by the vortical family, so no
  Nyquist residual lands here); on the channel the physical
  families sum to the Leray projector ``P`` (NOT the identity), so
  the complement captures exactly the divergence
  (pressure-gradient) content plus any column the labeler left
  ``UNLABELED``.

Each is a :class:`~fridom.model.transforms.projection.ProjectionFactory`
with dual sources — ``VorticalProjection(em)`` from an explicit
eigenmodes object, or ``VorticalProjection.from_model(model,
at_time=...)`` (which dispatches on the grid topology).

Staggered spectral basis (the analytic path). The nonhydro modes are
the *discrete* C-grid eigenvectors, and the projection is a clean
per-component round-trip through the grid's own transforms: each
``(u, v, w, b)`` component is forward-transformed on its **own**
physical space (``u``/``v``/``w`` face-staggered, ``b`` collocated)
via the eigenmode kit (``em.kit.forward(name)``),
``sum_s em.projector(s)`` acts on the coefficient state (diagonal
per wavenumber — the half-cell staggering lives in the eigenvectors'
operator symbols), and the backward transforms return to the
staggered physical spaces. The coefficient layout is the grid's
real-FFT layout — half spectrum on the first transformed axis,
full-complex later axes — so on a real state a single-branch
projection is Hermitian-closed by the real part of the backward
transform (see :func:`mode_projection`). The transforms carry the
concrete staggered endo ``StateSignature`` over ``(u, v, w, b)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.eigenbasis import (
    family_projection,
)
from fridom.model.eigenbasis import (
    predicate_projection as predicate_projection,  # noqa: PLC0414 — re-export
)
from fridom.model.transforms.projection import (
    EigenProjection,
    ProjectionFactory,
)
from fridom.model.transforms.signature import StateSignature
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.eigenmodes import from_model
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.transforms.base import StateTransform
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


def _build_vortical(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the geostrophic / labeled-vortical projection.

    On the channel this is the labeled **vortical** family only: the
    divergence-complement zero modes carry the separate
    ``constraint`` label and are provably excluded (the zero-space
    overlap split rotates the columns apart before labeling).
    """
    if isinstance(em, ChannelEigenmodes):
        return family_projection(em, "vortical",
                                 name="VorticalProjection")
    return mode_projection(em, 0, name="VorticalProjection")


def _build_wave(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the inertia-gravity projection (both branches, merged).

    On the channel this is the labeled wave family only (``wave+``
    and ``wave-``): Kelvin is a separate family
    (:data:`KelvinProjection`), not part of the wave selection.
    """
    if isinstance(em, ChannelEigenmodes):
        return family_projection(em, "wave", name="WaveProjection")
    return mode_projection(em, 1) + mode_projection(em, -1)


def _build_kelvin(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the boundary-trapped Kelvin projection (channel only)."""
    if isinstance(em, ChannelEigenmodes):
        return family_projection(em, "kelvin",
                                 name="KelvinProjection")
    raise ValueError(
        "no walls, no Kelvin family: Kelvin modes are trapped at "
        "walls the rotation couples to, and the analytic nonhydro "
        "eigenmodes (fully periodic or walled-vertical grid) carry "
        "only the vortical and inertia-gravity branches. Build the "
        "model on a horizontally walled channel grid to get the "
        "labeled Kelvin pair")


def _build_divergence(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the residual complement of every named physical family.

    On the analytic tier the complement carries the divergence
    (pressure-gradient) content plus the structurally excluded
    ``k_h = 0`` inertial strata and the ``k = 0`` mean; the
    even-grid Nyquist steady strata belong to the vortical family
    and never land here. On the channel the named physical families
    are vortical, wave and kelvin, which together sum to the Leray
    projector ``P`` (NOT the identity — the ``constraint`` columns
    are excluded), so the complement captures the divergence
    content plus any column the labeler left ``UNLABELED`` — no
    unresolved column ever lands in a named family silently.
    """
    named = _build_vortical(em) + _build_wave(em)
    if isinstance(em, ChannelEigenmodes):
        named = named + _build_kelvin(em)
    return named.complement


# ================================================================
#  Public dual-source projections
# ================================================================
VorticalProjection = ProjectionFactory(
    _build_vortical, from_model, "VorticalProjection")
WaveProjection = ProjectionFactory(
    _build_wave, from_model, "WaveProjection")
KelvinProjection = ProjectionFactory(
    _build_kelvin, from_model, "KelvinProjection")
DivergenceProjection = ProjectionFactory(
    _build_divergence, from_model, "DivergenceProjection")
