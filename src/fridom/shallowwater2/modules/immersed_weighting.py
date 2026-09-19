r"""Immersed cut-cell weighting idioms for the shallow-water physics.

Description
-----------
The fraction/mask weighting shared by the linear ``gravity`` core term
and the nonlinear ``SadournyAdvection`` transport on immersed
(cut-cell) grids (IP-D10). Three idioms, mirroring the
``_immersed_flux`` / ``_immersed_scale`` machinery of
``fr.model.modules.advection`` (I2) but implemented locally in
shallowwater2 — the model package owns its bespoke reductions (IP-D4):

- :func:`weight_flux` weights a face flux by the open-area fraction
  :math:`\alpha_f`, so a closed face (:math:`\alpha = 0`) carries no
  flux and, on a face-aligned :math:`\{0, 1\}` staircase, reproduces
  the walled model;
- :func:`scale_divergence` divides a flux divergence by the wet
  plan-area fraction :math:`\theta_c` (guarded: a dry cell stays
  exactly ``0``), so ``sum_c theta_c V_c q_c`` is conserved to machine
  zero (the flux differences telescope over the wet region);
- :func:`mask_field` zeros a field on its dry DOFs with the boolean
  per-space mask (the immersed free-slip / impermeability closure).

Every idiom fetches its fraction/mask on the field's own (possibly
wall-tagged) space, so the multiply is a plain same-space product.
Callers guard on ``grid.immersed is not None`` and never enter off an
immersed grid, so these helpers stay a pure no-op on the unimmersed
path (the parity guard).

On a chart grid (the masked sphere) the same three idioms compose with
the metric path for the full-cell staircase only;
:func:`require_chart_composable` is the shared bind-time refusal of
the combinations that do not compose (a non-orthogonal chart, genuine
partial cells).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.immersed_domain import ImmersedDomain


#: the smallest quadrature order that yields genuine partial cells
_PARTIAL_ORDER = 2


def require_chart_composable(grid: Grid, owner: str) -> None:
    r"""
    Refuse the chart + immersed combinations that do not compose.

    Description
    -----------
    The chart + immersed composition of the shallow-water physics
    (the masked sphere) is deliberately narrow — an **orthogonal**
    chart carrying a full-cell **staircase** mask:

    - **Orthogonal chart.** The fraction/mask weights sit on the
      direct face fluxes and the finished momentum tendencies. A
      non-orthogonal chart couples the components through the
      cross-metric hops of ``raise_index``; keeping those skew under a
      mask needs the fraction *between* the hop pairs (the MI-D2
      spelling of the mapped+immersed composition plan), which the
      shallow-water terms do not carry.
    - **Staircase.** The collocation ``ImmersedDomain``
      (``order=None``) samples the indicator at the cell centres in
      the chart coordinates; every fraction is an exact ``0`` or ``1``
      and the metric weights cancel out of it. A quadrature order
      ``>= 2`` averages the indicator per axis in the **chart
      parameters**, not over the physical area
      (:math:`\theta \ne \int\sqrt g\,\chi / \int\sqrt g`), so the
      fractions would be inconsistent with the metric the operators
      consume — the geometric-consistency (freestream) trap.

    A no-op off a chart grid and off an immersed grid.

    Parameters
    ----------
    grid : Grid
        The grid the module binds to.
    owner : str
        The refusing module's name, for the message.

    Raises
    ------
    NotImplementedError
        If the grid carries an immersed domain together with a
        non-orthogonal chart, or with genuine partial cells on a chart.
    """
    immersed = getattr(grid, "immersed", None)
    if grid.chart_coords is None or immersed is None:
        return
    if not grid.mapping.orthogonal:
        raise NotImplementedError(
            f"{owner} supports an immersed domain on a chart grid "
            "only when the chart is declared orthogonal "
            "(CoordinateMapping(..., orthogonal=True), e.g. the "
            "lat-lon sphere): on a non-orthogonal chart the "
            "cross-metric terms couple the velocity components "
            "through interpolation hops the mask does not reach, so "
            "the masked wave operator would lose its skew-symmetry "
            "(energy leaks at the wet-region boundary). Drop the "
            "immersed domain, or use an orthogonal chart.")
    order = immersed.order
    if order is not None and order >= _PARTIAL_ORDER:
        raise NotImplementedError(
            f"{owner} supports an immersed domain on a chart grid "
            "(e.g. a land mask on the lat-lon sphere) only as the "
            "full-cell staircase, ImmersedDomain(..., order=None): "
            f"order={order} asks for genuine partial cells, whose "
            "per-axis quadrature averages the indicator in the chart "
            "parameters rather than over the physical area, so the "
            "fractions would be inconsistent with the metric (a "
            "spurious pressure gradient in a resting ocean). Declare "
            "the mask with order=None; area-weighted chart fractions "
            "for shallowwater2 are a recorded follow-up.")


def weight_flux(
    immersed: ImmersedDomain, flux: ScalarField,
) -> ScalarField:
    r"""
    Weight one face flux by the open-area fraction :math:`\alpha_f`.

    Description
    -----------
    ``F <- alpha_f * F`` with ``alpha_f = fraction(flux space)`` (the
    min-rule face fraction of I0) — the cut-cell area weight that
    closes an :math:`\alpha = 0` face as a free no-normal-flow wall.
    The fraction is fetched on the flux's own space, so the multiply is
    a plain same-space product.

    Parameters
    ----------
    immersed : ImmersedDomain
        The grid's immersed descriptor.
    flux : ScalarField
        The advective flux on a control-volume face.

    Returns
    -------
    ScalarField
        The open-area-weighted flux.
    """
    return flux * immersed.fraction(flux.function_space)


def scale_divergence(
    immersed: ImmersedDomain, res: ScalarField,
) -> ScalarField:
    r"""
    Divide a flux divergence by the wet plan-area fraction (guarded).

    Description
    -----------
    The masked tendency ``(1/(theta_c V_c)) sum_f +/- alpha_f A_f F_f``:
    the summed open-area-weighted flux divergence (already
    ``1/V_c``-scaled by ``diff``) is divided by the cell plan-area
    fraction ``theta_c = fraction(res space)``. Guarded so a dry cell
    (``theta = 0``, numerator identically ``0``) stays exactly ``0``.
    The ``theta``-weighted tendency conserves ``sum_c theta_c V_c q_c``
    to machine zero.

    Parameters
    ----------
    immersed : ImmersedDomain
        The grid's immersed descriptor.
    res : ScalarField
        The accumulated flux divergence on the codomain (cell) space.

    Returns
    -------
    ScalarField
        The wet-plan-area-scaled tendency.
    """
    theta = immersed.fraction(res.function_space)
    wet = theta.data > 0.0
    scaled = jnp.where(
        wet, res.data / jnp.where(wet, theta.data, 1.0), 0.0)
    return res.with_data(scaled)


def mask_field(
    immersed: ImmersedDomain, field: ScalarField,
) -> ScalarField:
    r"""
    Zero a field on its dry DOFs (boolean per-space wet mask).

    Description
    -----------
    ``phi <- phi * mask(phi space)`` with the boolean staggered mask
    ``theta > 0`` under the domain's slip rule — the immersed
    free-slip (a corner vorticity touching a dry cell is zeroed) and
    impermeability (no tendency into a closed face) closure.

    Parameters
    ----------
    immersed : ImmersedDomain
        The grid's immersed descriptor.
    field : ScalarField
        The field (velocity tendency, corner PV, ...) to mask.

    Returns
    -------
    ScalarField
        The field with its dry DOFs set to exactly ``0``.
    """
    mask = immersed.mask(field.function_space)
    return field.with_data(field.data * mask.data)
