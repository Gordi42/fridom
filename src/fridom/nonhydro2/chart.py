r"""Chart-native derivation for the nonhydrostatic state (ruling (d)).

Description
-----------
The per-package derivation hook behind ``nh.State.chart``
(``physical_state_components.md`` ruling (d)). nonhydro2 stores the
**physical** velocity trio ``u, v, w`` on every grid — mapped columns
included, where ``w`` is the prognostic physical vertical velocity —
so the horizontal components and the flat / unmapped ``w`` are already
chart-native (the identity). On a terrain-following (sigma) column the
chart-native vertical quantity is the **contravariant volume flux**

.. math::

    J\omega = w - \sum_i Z_i\, I(u_i) ,

with ``Z_i = d<mapped>_d<axis>`` the coordinate-surface slopes and
``I(u_i)`` the coupled physical velocity interpolated onto the ``w``
face. This is exactly the quantity the mapped pressure solver's
divergence right-hand side derives
(``mapped_pressure.MappedPressureSolver.divergence``:
``sum_i D_i(J u_i) + D_b(w - sum_i Z_i I(u_i))``) — the boundary-
conforming flux whose column difference closes the walls with zero
normal flow. Deriving it here (read-only, on demand) keeps ``w`` the
physical velocity everywhere while exposing the flux the solver
measures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField
    from fridom.spatial.grid import Grid


def _vertical_column(grid: Grid) -> tuple[str, str] | None:
    r"""Return the ``(mapped, base)`` vertical sigma column, or ``None``.

    Description
    -----------
    The single-base analytic column whose base coordinate is itself
    (``column_corrections[axis][1] == axis`` — the vertical column, as
    opposed to a coupled horizontal coordinate that references it).
    ``None`` off a mapped grid or on an embedding chart, where the
    stored ``w`` is already chart-native and the view is the identity.
    A read-only diagnostic, so it never raises.

    Parameters
    ----------
    grid : Grid
        The bound grid.

    Returns
    -------
    tuple[str, str] | None
        The ``(mapped, base)`` vertical column pair, or ``None``.
    """
    mapping = getattr(grid, "mapping", None)
    if mapping is None or mapping.chart_coords is not None:
        return None
    for axis, entry in mapping.column_corrections.items():
        if entry[1] == axis:
            return entry
    return None


def chart_component(state: VectorField, name: str) -> ScalarField:
    r"""Return a nonhydrostatic state component in its chart-native form.

    Description
    -----------
    The identity for every component except the vertical velocity ``w``
    on a terrain (sigma) column, which becomes the **contravariant
    volume flux** ``J\omega = w - \sum_i Z_i I(u_i)`` — the stored
    physical ``w`` minus the coupled velocities (``u`` along the first
    horizontal axis, ``v`` along the second) interpolated onto the ``w``
    face and weighted by the slope metrics ``Z_i = d<mapped>_d<axis>``.
    On an unmapped grid, on an embedding chart and for the horizontal /
    tracer components the stored physical field is already the
    chart-native quantity, so the identity is returned.

    Parameters
    ----------
    state : VectorField
        The nonhydrostatic state (physical components).
    name : str
        The component name to view.

    Returns
    -------
    ScalarField
        The chart-native component (the flux for mapped ``w``, else the
        stored physical field).
    """
    w = state[name]
    if name != "w":
        return w
    grid = w.grid
    column = _vertical_column(grid)
    if column is None:
        return w
    mapped, base = column
    horizontal = tuple(a for a in grid.names if a != base)
    bare = w.function_space.bare
    flux = w
    for comp, axis in zip(("u", "v"), horizontal, strict=True):
        iu = state[comp].to(w)
        zi = grid.metric(bare, f"d{mapped}_d{axis}")
        flux = flux - zi.retag(iu) * iu
    return flux
