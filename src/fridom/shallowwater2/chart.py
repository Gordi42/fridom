r"""Chart seam conversions and chart-native derivation (shallowwater2).

Description
-----------
The shallowwater2 physical/contravariant seam
(``physical_state_components.md`` rulings (c) + (d);
``sw2_physical_flip_plan.md`` D1/D2/D5/D6). The user-facing state
components ``u``, ``v`` are the **physical** (m/s) velocities on every
grid — flat and spherical alike (invariant (a)) — while the
metric-aware chart terms are written for the **contravariant**
coordinate velocities :math:`u^i = \dot\lambda,\ \dot\varphi`. This
module owns the two seam conversions and the ``state.chart`` derivation
hook:

- **entry** (:func:`to_contravariant`): :math:`u^i = U_i /
  \sqrt{g_{ii}}`, a VJP-sealed divide (D2) on the component's **own**
  bare staggered space (D6);
- **exit** (:func:`to_physical_tendency`): :math:`\mathrm{d}U_i =
  \sqrt{g_{ii}}\,\mathrm{d}u^i`, a plain multiply (the metric is
  static, so :math:`\mathrm{d}/\mathrm{d}t` commutes with the
  rescale);
- :func:`chart_component`: the ``state.chart`` hook (ruling (d)) — the
  coordinate velocity :math:`\dot\lambda = U/\sqrt{g_{\lambda\lambda}}`
  on a chart grid, the identity on flat grids and for ``p``.

Everything **between** the seams runs the existing contravariant
spellings verbatim (D1): the conversion is the similarity transform
:math:`T_{\rm phys} = S\,T_{\rm con}\,S^{-1}` with
:math:`S = \mathrm{diag}(\sqrt{g_{ii}})` static and pointwise, so every
energy-exactness proof conjugates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.fields.scalar_field import ScalarField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.vector_field import VectorField


def _sealed_metric_divide(
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
    The same double-``where`` seal as ``sadourny._sealed_divide``. It
    runs on real storage only — the seam sits inside the halo-trace
    exempt chart terms and the read-only ``state.chart`` view, never
    with storage-less operands.
    """
    guarded = jnp.where(den.storage == 0.0, 1.0, den.storage)
    safe = ScalarField(
        den.grid, den.function_space, guarded,
        den.metadata, halo_valid=den.halo_valid)
    return num / safe


def to_contravariant(field: ScalarField, axis: str) -> ScalarField:
    r"""Return the contravariant component ``U_i / sqrt(g_ii)`` (entry).

    Description
    -----------
    The seam entry conversion (D1/D2/D6): the physical component
    :math:`U_i` becomes the contravariant coordinate velocity
    :math:`u^i = U_i / \sqrt{g_{ii}}`, with :math:`g_{ii}` derived on
    the component's **own** bare staggered space via ``grid.metric``
    (never cached) and the divide VJP-sealed
    (:func:`_sealed_metric_divide`). The result carries the component's
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
    return _sealed_metric_divide(field, root)


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


def chart_component(state: VectorField, name: str) -> ScalarField:
    r"""Return a shallow-water state component in its chart-native form.

    Description
    -----------
    The per-package ``state.chart`` derivation hook (ruling (d), the
    ``ChartView`` + ``nonhydro2/chart.py`` pattern, CS-14-safe). On a
    chart grid ``chart["u"]`` / ``chart["v"]`` are the **coordinate
    velocities** :math:`\dot\lambda = U/\sqrt{g_{\lambda\lambda}}`,
    :math:`\dot\varphi = V/\sqrt{g_{\varphi\varphi}}` (the entry
    conversion, sealed divide). On an unmapped grid, and for the tracer
    component ``p`` on any grid, the stored physical field is already
    the chart-native quantity, so the identity is returned.

    Parameters
    ----------
    state : VectorField
        The shallow-water state (physical components).
    name : str
        The component name to view.

    Returns
    -------
    ScalarField
        The chart-native component (the coordinate velocity for a chart
        ``u`` / ``v``, else the stored physical field).
    """
    field = state[name]
    grid = field.grid
    if grid.chart_coords is None or name not in ("u", "v"):
        return field
    names = field.function_space.names
    axis = names[0] if name == "u" else names[1]
    return to_contravariant(field, axis)
