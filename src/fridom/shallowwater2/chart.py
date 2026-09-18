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

The two conversions (and the sealed divide) live in the shared home
:mod:`fridom.model.chart_seams` (spherical-models plan, SP-D7 — the
3-D models reuse them verbatim) and are re-exported here under their
shallow-water spellings.

Everything **between** the seams runs the existing contravariant
spellings verbatim (D1): the conversion is the similarity transform
:math:`T_{\rm phys} = S\,T_{\rm con}\,S^{-1}` with
:math:`S = \mathrm{diag}(\sqrt{g_{ii}})` static and pointwise, so every
energy-exactness proof conjugates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.chart_seams import (
    sealed_metric_divide as _sealed_metric_divide,  # noqa: F401 — re-export
)
from fridom.model.chart_seams import (
    to_contravariant,
    to_physical_tendency,  # noqa: F401 — the seam's sw2 spelling
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField


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
