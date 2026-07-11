r"""
Shallow-water eigenmode projections (``sw.transforms``, wave 7 C).

Description
-----------
The Tier-1 vortical / wave / kelvin / divergence projections as
composable ``fr.StateTransform``s, with **dual eigenmode backends**
routed on the eigenmodes object type:

- the analytic operator-sourced ``sw.eigenmodes.Eigenmodes`` of the
  fully periodic grid (per-mode spectral projector, the original
  path, bitwise unchanged);
- the labeled numeric ``sw.ChannelEigenmodes`` of the walled channel
  (the dense-column engine path: partial-axis Fourier along the
  periodic axis, per-plane column projection under the energy
  metric).

The public factories:

- ``VorticalProjection`` — the geostrophic mode ``P(0)``
  (periodic), or the labeled **vortical** family (channel);
- ``WaveProjection`` — the inertia-gravity modes ``P(+1) + P(-1)``
  (periodic), or the labeled **wave+ and wave-** Poincaré branches
  only (channel; Kelvin is a separate family);
- ``KelvinProjection`` — the boundary-trapped **kelvin+ and
  kelvin-** pair (channel only; the fully periodic grid raises: no
  walls, no Kelvin family);
- ``DivergenceProjection`` — the residual complement of every named
  family. On the channel it is the zero map up to floating point
  wherever the labeler resolved every column (the f-plane channel);
  columns the labeler left ``UNLABELED`` never land in a named
  family silently — the complement captures exactly them.

Each is a :class:`~fridom.framework2.transforms.projection.ProjectionFactory`
with **dual sources** — ``VorticalProjection(em)`` from an explicit
eigenmodes object, or ``VorticalProjection.from_model(model,
at_time=...)`` (which dispatches on the grid topology).

Staggered coefficient basis (the analytic path). The shallow-water
modes are the *discrete* C-grid eigenvectors, so the projection is a
clean round-trip on the model's physical **staggered** state (``u``
on the east face, ``v`` on the north face, ``p`` on the centre): the
eigenmode kit's per-component transforms carry each component to its
own coefficient basis, the diagonal per-mode projector applies
there, and the backward transforms return to the nodal spaces. The
three modes span the three components at every wavenumber (the
``k = 0`` mean via the inertial patch), except the
interpolation-Nyquist planes where the geostrophic column is a
structural zero — ``DivergenceProjection`` picks up exactly that
Nyquist-vortical residual and is the zero map on Nyquist-free
states.

Per-plane column projection (the engine path). The shared framework
machinery
(:mod:`fridom.framework2.model.eigenbasis` — re-exported here as
:func:`family_projection` / :func:`predicate_projection`): each
component is forward-transformed along the **periodic axis only**
(its own partial-axis Fourier space, the bounded axis stays nodal),
the component segments concatenate to one stacked column per
``rfft`` half-spectrum plane, and the plane projector ``P = q_sel
(q_sel^H M z)`` applies with the label mask built from the requested
family codes — reading ``q``/``labels`` off the
:class:`ChannelEigenmodes` at application time (after labeling; the
Kelvin degeneracy recovery may have rotated ``q`` in place). Every
named-family selection is closed under conjugation (vortical is
self-conjugate, wave and kelvin carry both signed branches), so the
real backward synthesis is exact — the implied negative-``kx``
planes carry the conjugate selection and the result is exactly real
up to floating point. The whole path is jax-traceable and
sharding-clean: the plane projector is an einsum of the (replicated)
basis against the coefficient planes, whose sharding flows through
the grid's transform and storage contracts (no host gather).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework2 as fr
from fridom.framework2.model.eigenbasis import (
    family_projection,
)
from fridom.framework2.model.eigenbasis import (
    predicate_projection as predicate_projection,  # noqa: PLC0414 — re-export
)
from fridom.framework2.transforms.projection import (
    EigenProjection,
    ProjectionFactory,
)
from fridom.framework2.transforms.signature import StateSignature
from fridom.shallowwater2.channel_eigenmodes import ChannelEigenmodes
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
    """Return the staggered endo signature ``(u, v, p)``.

    The analytic eigenmodes resolve the canonical C-grid stagger on
    their fully periodic (tag-free) grid; the channel eigenmodes
    carry the model's own (BC-tagged) component spaces through the
    shared framework projections instead.
    """
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


def _build_vortical(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the geostrophic / labeled-vortical projection."""
    if isinstance(em, ChannelEigenmodes):
        return family_projection(em, "vortical",
                                 name="VorticalProjection")
    return mode_projection(em, 0, name="VorticalProjection")


def _build_wave(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the inertia-gravity projection (both branches, merged).

    On the channel this is the labeled **Poincaré** family only
    (``wave+`` and ``wave-``): Kelvin is a separate family
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
        "no walls, no Kelvin family: Kelvin modes are boundary-"
        "trapped, and the fully periodic shallow-water eigenmodes "
        "carry only the vortical and inertia-gravity branches. "
        "Build the model on a channel grid (exactly one bounded "
        "axis) to get the labeled Kelvin pair")


def _build_divergence(
    em: Eigenmodes | ChannelEigenmodes,
) -> StateTransform:
    """Return the residual complement of every named family.

    On the channel the named families are vortical, wave (both
    Poincaré branches) and kelvin (both branches); the complement is
    the zero map up to floating point wherever the labeler resolved
    every column (the f-plane channel) and captures exactly the
    ``UNLABELED`` columns otherwise — no unresolved column ever
    lands in a named family silently.
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
