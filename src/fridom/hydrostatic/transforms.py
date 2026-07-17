r"""
Hydrostatic eigenmode projections (``hy.transforms``, HY-D7).

Description
-----------
The Tier-1 vortical / wave and barotropic / baroclinic projections as
composable ``fr.model.StateTransform``s, backed by the numeric
:class:`~fridom.hydrostatic.eigenmodes.HydrostaticEigenmodes` (the
dense-column channel engine on the bounded vertical). Each is a
:class:`~fridom.model.transforms.projection.ProjectionFactory` with dual
sources — ``VorticalProjection(em)`` from an explicit eigenmodes object,
or ``VorticalProjection.from_model(model, at_time=...)`` (which builds
the eigenmodes first, refusing the implicit / rigid-lid variant through
``require_linear_operator``).

The public factories:

- ``VorticalProjection`` — the geostrophic (steady) modes: the
  barotropic and baroclinic ``vortical`` families;
- ``WaveProjection`` — every inertia-gravity mode: the barotropic and
  baroclinic ``wave+``/``wave-`` families;
- ``BarotropicProjection`` — the free-surface (depth-mean) branch: the
  ``barotropic_*`` families (geostrophic + the surface-pressure wave
  pair);
- ``BaroclinicProjection`` — the internal branch: the ``baroclinic_*``
  families (geostrophic + the internal-wave pairs).

The two classification axes are orthogonal and complete:
``Vortical + Wave == Barotropic + Baroclinic == identity`` on the linear
subspace (every column is labeled), so ``project -> reconstruct`` is
exact on a random linear-subspace state. The projections merge by label
code union under ``+`` (``EigenProjection.__add__``), so
``.complement`` stays a valid idempotent projector.

The per-plane column projection ``Q diag(m) Q^H M z`` runs under the
hydrostatic energy metric ``diag(1, 1, 1/N^2, 1/c^2)`` (the ``ps``
weight depth-integrated), the metric the eigenvectors are M-orthonormal
under, so the summed family projectors are orthogonal projectors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.hydrostatic.eigenmodes import from_model
from fridom.model._eigenbasis import family_projection
from fridom.model.transforms.projection import ProjectionFactory

if TYPE_CHECKING:  # pragma: no cover
    from fridom.hydrostatic.eigenmodes import HydrostaticEigenmodes
    from fridom.model.transforms.base import StateTransform


# ================================================================
#  Composite builders (label-code union under ``+``)
# ================================================================
def _build_vortical(em: HydrostaticEigenmodes) -> StateTransform:
    """Return the geostrophic projection (barotropic + baroclinic)."""
    return (
        family_projection(em, "barotropic_vortical")
        + family_projection(em, "baroclinic_vortical"))


def _build_wave(em: HydrostaticEigenmodes) -> StateTransform:
    """Return the inertia-gravity projection (all four wave branches)."""
    return (
        family_projection(em, "barotropic_wave")
        + family_projection(em, "baroclinic_wave"))


def _build_barotropic(em: HydrostaticEigenmodes) -> StateTransform:
    """Return the barotropic (free-surface) branch projection."""
    return (
        family_projection(em, "barotropic_vortical")
        + family_projection(em, "barotropic_wave"))


def _build_baroclinic(em: HydrostaticEigenmodes) -> StateTransform:
    """Return the baroclinic (internal) branch projection."""
    return (
        family_projection(em, "baroclinic_vortical")
        + family_projection(em, "baroclinic_wave"))


# ================================================================
#  Public dual-source projections
# ================================================================
VorticalProjection = ProjectionFactory(
    _build_vortical, from_model, "VorticalProjection")
WaveProjection = ProjectionFactory(
    _build_wave, from_model, "WaveProjection")
BarotropicProjection = ProjectionFactory(
    _build_barotropic, from_model, "BarotropicProjection")
BaroclinicProjection = ProjectionFactory(
    _build_baroclinic, from_model, "BaroclinicProjection")
