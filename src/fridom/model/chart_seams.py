r"""Chart seam conversions shared by the chart-capable models.

Description
-----------
The physical/contravariant **seam** of every model that runs on an
embedding chart (``physical_state_components.md`` rulings (a)-(d);
spherical-models plan SP-D7): the stored velocity components are the
**physical** (m/s) velocities on every grid, while metric-aware chart
terms are written for the contravariant coordinate velocities
:math:`u^i = U_i/\sqrt{g_{ii}}`. This module is the shared home of

- the two seam conversions :func:`to_contravariant` (entry, a
  VJP-sealed divide) and :func:`to_physical_tendency` (exit, a plain
  multiply) — promoted verbatim from ``shallowwater2.chart``, which
  re-exports them;
- the **orthogonal thin-shell** helpers the 3-D flux-form models are
  written in (SP-D1 / SP-D2 / SP-D6 fallback): on an orthogonal
  two-coordinate chart :math:`(x^1, x^2)` extruded along a flat
  vertical, with scale factors :math:`h_i = \sqrt{g_{ii}}` and
  :math:`\sqrt g = h_1 h_2` (independent of the vertical), the
  divergence of a vector with **physical** components is

  .. math::
      \nabla\cdot\vec F = \frac{1}{\sqrt g}\bigl[
          \partial_1(h_2 F_1) + \partial_2(h_1 F_2)\bigr]
          + \partial_z F_z ,

  and the physical gradient component is
  :math:`(\nabla p)_i = \partial_i p / h_i`.
  :func:`edge_scale` is the transverse edge length :math:`h_j` that
  weights a face flux, :func:`volume_scale` the cell area
  :math:`\sqrt g`, :func:`scale_factor` the :math:`h_i` of the
  gradient. On the identity chart every factor is exactly 1.0, so the
  chart spelling reduces to the flat one bitwise (``x * 1.0 == x``,
  ``x / 1.0 == x``).

Every metric is derived through ``grid.metric`` on the operand's own
bare staggered space at application — never cached (rules 2.3/3.8).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.fields.scalar_field import ScalarField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.grid import Grid


# ================================================================
#  Chart discovery (taught errors)
# ================================================================
def thin_shell_chart(grid: Grid, consumer: str) -> tuple[str, str] | None:
    r"""Return the orthogonal 2-coordinate chart pair, or ``None``.

    Description
    -----------
    ``None`` off a chart grid (the byte-identical flat / ``maps=``
    paths). On a chart grid the thin-shell flux-form spelling needs an
    **orthogonal** chart on exactly **two** coordinates with no
    ``maps=`` column on top (chart + terrain(sigma) is a later
    composition, spherical-models plan section 6): anything else is a
    taught error, never a silently metric-blind run.

    Parameters
    ----------
    grid : Grid
        The bound grid.
    consumer : str
        The refusing module's name (for the message).

    Returns
    -------
    tuple[str, str] | None
        The chart coordinate names in grid factor order, or None.

    Raises
    ------
    NotImplementedError
        For a non-orthogonal chart, a chart on other than two
        coordinates, or a chart combined with a mapped column.
    """
    chart = getattr(grid, "chart_coords", None)
    if chart is None:
        return None
    mapping = grid.mapping
    pair = tuple(name for name in grid.names if name in set(chart))
    if (len(pair) != 2  # noqa: PLR2004 — a surface chart
            or not mapping.orthogonal or mapping.column_corrections):
        raise NotImplementedError(
            f"{consumer} supports embedding charts in the orthogonal "
            "thin-shell form only: an orthogonal=True chart on exactly "
            "two coordinates (e.g. fr.spatial.charts.lonlat_sphere / "
            "torus) extruded along a flat vertical, with no maps= "
            f"column on top. This grid's chart is on {tuple(chart)} "
            f"(orthogonal={mapping.orthogonal}, mapped columns: "
            f"{sorted(mapping.column_corrections)}). Non-orthogonal "
            "charts and chart + terrain(sigma) compositions are "
            "future work (spherical-models plan, section 6)")
    return pair


# ================================================================
#  Orthogonal thin-shell geometry factors
# ================================================================
def scale_factor(field: ScalarField, axis: str) -> ScalarField:
    r"""Return :math:`h_{axis} = \sqrt{g_{axis,axis}}` on ``field``'s space."""
    return field.grid.metric(
        field.function_space.bare, f"g_{axis}{axis}") ** 0.5


def edge_scale(
    field: ScalarField, axis: str, chart: tuple[str, str],
) -> ScalarField | None:
    r"""Return the transverse edge length weighting a flux along ``axis``.

    Description
    -----------
    For a flux along chart coordinate :math:`x^i` the face "area" per
    unit computational cross-section is the scale factor of the
    **other** chart coordinate, :math:`h_j` (:math:`\sqrt g / h_i` on
    an orthogonal chart), derived on the flux's own staggered space. A
    flux along a non-chart (flat vertical) axis is weighted by
    :math:`\sqrt g`, which cancels against the volume division because
    the thin-shell :math:`\sqrt g` does not depend on the vertical — so
    ``None`` is returned and the caller skips both factors.

    Parameters
    ----------
    field : ScalarField
        The flux (its space fixes the staggering of the weight).
    axis : str
        The flux axis.
    chart : tuple[str, str]
        The chart coordinate pair (:func:`thin_shell_chart`).

    Returns
    -------
    ScalarField | None
        :math:`h_j` on the flux space, or None off the chart axes.
    """
    if axis not in chart:
        return None
    other = chart[1] if axis == chart[0] else chart[0]
    return scale_factor(field, other)


def volume_scale(field: ScalarField) -> ScalarField:
    r"""Return the cell area :math:`\sqrt g` on ``field``'s own space."""
    return field.grid.metric(field.function_space.bare, "sqrt_g")


def chart_gradient(
    field: ScalarField, axis: str,
) -> ScalarField:
    r"""Return the physical gradient component ``d_axis(field) / h_axis``.

    Description
    -----------
    The staggered difference of a scalar along a chart coordinate,
    divided by the scale factor on the **difference's** own face space
    (the VJP-sealed metric divide — the metric root is an exact zero in
    the never-valid padding). On the identity chart the divisor is
    exactly 1.0 (bitwise the plain ``diff``).
    """
    diff = field.diff(axis)
    return sealed_metric_divide(diff, scale_factor(diff, axis))


# ================================================================
#  The seam conversions (promoted from shallowwater2.chart)
# ================================================================
def sealed_metric_divide(
    num: ScalarField, den: ScalarField,
) -> ScalarField:
    r"""Return ``num / den`` with the metric denominator VJP-sealed (D2).

    Description
    -----------
    The seam entry divide :math:`U_i / \sqrt{g_{ii}}`. On a walled chart
    (the lat-lon sphere's polar caps) the metric root is an **exact
    zero** in the never-valid storage/halo padding, so the bare quotient
    both plants a forward pad-``inf`` (the ``mapped_chunk_nonfinite``
    failure class) and, in reverse mode, turns the zero cotangent of a
    stripped cell into ``0 * inf = NaN`` and poisons every gradient with
    a data path through the seam (AGENTS.md differentiability policy).
    Replacing each **exact-zero** denominator by 1 before the quotient
    keeps the ratio finite forward and reverse; valid cells
    (``den != 0``) divide by the true metric and are bitwise unchanged.
    The same double-``where`` seal as ``sadourny._sealed_divide``. When
    the metric lives on a **reduced** space (the 3-D thin shell: the
    metric is constant along the vertical) the quotient is formed as
    ``num * (1 / den)`` with the sealed reciprocal, because the
    algebra's lift of the denominator onto the numerator's space
    re-pads it with zeros. It
    runs on real storage only — the seam sits inside the halo-trace
    exempt chart terms and the read-only ``state.chart`` view, never
    with storage-less operands.
    """
    guarded = jnp.where(den.storage == 0.0, 1.0, den.storage)
    safe = ScalarField(
        den.grid, den.function_space, guarded,
        den.metadata, halo_valid=den.halo_valid)
    if den.function_space.bare == num.function_space.bare:
        return num / safe
    # the metric lives on a reduced space (the thin-shell metric is
    # constant along the vertical): the field algebra lifts it onto the
    # numerator's space and RE-PADS the lifted operand with exact
    # zeros, which would undo the seal (forward pad-inf, reverse NaN).
    # Multiply by the sealed reciprocal instead — a zero re-pad of the
    # reciprocal zeros the padding, finite forward and reverse. On the
    # identity chart the reciprocal is exactly 1.0 (bitwise neutral).
    return num * (1.0 / safe)


def to_contravariant(field: ScalarField, axis: str) -> ScalarField:
    r"""Return the contravariant component ``U_i / sqrt(g_ii)`` (entry).

    Description
    -----------
    The seam entry conversion (D1/D2/D6): the physical component
    :math:`U_i` becomes the contravariant coordinate velocity
    :math:`u^i = U_i / \sqrt{g_{ii}}`, with :math:`g_{ii}` derived on
    the component's **own** bare staggered space via ``grid.metric``
    (never cached) and the divide VJP-sealed
    (:func:`sealed_metric_divide`). The result carries the component's
    own space, so the chart-term body downstream runs verbatim.

    Parameters
    ----------
    field : ScalarField
        The physical velocity component :math:`U_i`.
    axis : str
        The component's own coordinate name (``"lon"`` for ``u``,
        ``"lat"`` for ``v`` on the sphere).

    Returns
    -------
    ScalarField
        The contravariant coordinate velocity :math:`u^i`.
    """
    root = field.grid.metric(
        field.function_space.bare, f"g_{axis}{axis}") ** 0.5
    return sealed_metric_divide(field, root)


def to_physical_tendency(
    tendency: ScalarField, axis: str,
) -> ScalarField:
    r"""Return the physical tendency ``sqrt(g_ii) * du^i`` (exit).

    Description
    -----------
    The seam exit conversion (D1): the contravariant tendency
    :math:`\mathrm{d}u^i` becomes the physical tendency
    :math:`\mathrm{d}U_i = \sqrt{g_{ii}}\,\mathrm{d}u^i`, a plain
    multiply (the metric is static, so :math:`\mathrm{d}/\mathrm{d}t`
    commutes with the rescale; no seal is needed — a zero metric root
    in the never-valid padding zeros the increment, finite forward and
    reverse). :math:`g_{ii}` is derived on the tendency's own bare
    staggered space via ``grid.metric`` (never cached).

    Parameters
    ----------
    tendency : ScalarField
        The contravariant tendency :math:`\mathrm{d}u^i`.
    axis : str
        The component's own coordinate name.

    Returns
    -------
    ScalarField
        The physical tendency :math:`\mathrm{d}U_i`.
    """
    root = tendency.grid.metric(
        tendency.function_space.bare, f"g_{axis}{axis}") ** 0.5
    return root * tendency
