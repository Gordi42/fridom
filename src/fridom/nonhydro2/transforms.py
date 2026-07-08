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
eigenvectors: the projector acts on the full-complex Fourier transform
of the staggered ``(u, v, w, b)`` fields (the discrete ``one_hat`` /
``k_hat`` symbols carry the half-cell staggering). The transform is the
plain multi-dimensional FFT (``jnp.fft.fftn`` / ``ifftn``) of each
component — the grid's real-FFT collocated transform uses a different
(halved, collocated) layout, so the round-trip is done at package level
here rather than through ``grid.dispatch``. Signatures are polymorphic
in this iteration (a concrete staggered-spectral ``StateSignature`` is
deferred with the grid-level staggered transform).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.transforms.projection import (
    EigenProjection,
    ProjectionFactory,
)
from fridom.nonhydro2.eigenmodes import from_model

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.transforms.base import StateTransform
    from fridom.nonhydro2.eigenmodes import Eigenmodes
    from fridom.nonhydro2.state import State

#: the nonhydro prognostic components the projector acts on.
_COMPONENTS = ("u", "v", "w", "b")


# ================================================================
#  The staggered-spectral projector (fftn -> project -> ifftn)
# ================================================================
def _project(
    em: Eigenmodes, modes: tuple[int, ...], state: State,
) -> State:
    r"""
    Project a staggered physical state onto the span of ``modes``.

    Description
    -----------
    Forward-transforms each ``(u, v, w, b)`` component with a plain
    ``fftn`` (the discrete staggering lives in the eigenvectors),
    applies ``sum_s em.projector(s)`` on the resulting spectral dicts,
    and inverse-transforms back. Components outside the family (e.g.
    the pressure ``p``) pass through unchanged (``rest="pass"``).
    """
    fields = {c: jnp.fft.fftn(jnp.asarray(state[c].data))
              for c in _COMPONENTS}
    projected: dict | None = None
    for s in modes:
        contribution = em.projector(s)(fields)
        projected = (contribution if projected is None else
                     {c: projected[c] + contribution[c]
                      for c in contribution})
    return state.replace(**{
        c: state[c].with_data(jnp.fft.ifftn(projected[c]))
        for c in _COMPONENTS})


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
        signature=None,
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
