r"""Terrain-following (sigma-coordinate) support for the hydrostatic core.

Description
-----------
The hydrostatic model runs on a **terrain-following** column when its
grid carries a ``CoordinateMapping`` declaring a single-base analytic
map ``p = M(b, params)`` on the vertical mesh axis — the classic
sigma coordinate ``zp = z * H(x, y)`` (base ``z``, physical height
``zp``, column depth ``H``). On such a grid the vertical increment
carries the column Jacobian ``J = d<p>_d<b>`` and the coordinate
surfaces tilt with slope ``Z_i = d<p>_d<x_i>`` (rules 3.8; the same
``grid.metric`` rows the shared mapped advection and the mapped
pressure solver consume). This module is the single seam the core and
the free surface read to discover the column and name those rows —
nothing here materializes an array; the metrics derive on demand
through ``grid.metric`` at trace time.

Off a mapped grid (a flat mesh, or a stretched-only
``MappedIntervalMesh`` column whose stretching already rides
``grid.measure``) :func:`discover_column` returns ``None`` and every
terrain code path is skipped — the flat model stays byte-identical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.grid import Grid


def discover_column(
    grid: Grid, vertical: str,
) -> tuple[str, str] | None:
    r"""Return the ``(mapped, base)`` vertical column, or ``None``.

    Description
    -----------
    Inspects the grid's ``CoordinateMapping`` (rules 3.8) for a
    single-base analytic column whose base coordinate is the
    hydrostatic vertical mesh axis ``vertical``. Returns the
    ``(mapped physical name, base name)`` pair — e.g. ``("zp", "z")``
    for ``zp = z * H(x, y)`` — which the DIAGNOSE stages and the
    baroclinic pressure gradient turn into the metric-row names
    ``d<mapped>_d<...>`` they weight by.

    Returns ``None`` when the grid carries no ``CoordinateMapping``
    (a flat mesh, or a stretched-only ``MappedIntervalMesh`` whose
    stretching already rides ``grid.measure``) — the byte-identical
    flat path. Raises a taught error for a mapping the hydrostatic
    terrain support does not model: an embedding ``chart=`` (the
    curvilinear / spherical case, hydrostatic plan §7), or a vertical
    axis that is not the base of a single-base analytic column
    (a horizontally-mapped or multi-base map — the vertical integral
    has no single column Jacobian to carry).

    Parameters
    ----------
    grid : Grid
        The bound grid.
    vertical : str
        The hydrostatic vertical mesh-axis name (the axis the
        ``CumulativeIntegral`` reduces).

    Returns
    -------
    tuple[str, str] | None
        The ``(mapped, base)`` column pair, or ``None`` off a mapped
        grid.

    Raises
    ------
    NotImplementedError
        For an embedding chart, or a vertical axis that is not the
        single base of an analytic column.
    """
    mapping = getattr(grid, "mapping", None)
    if mapping is None:
        return None
    if mapping.chart_coords is not None:
        raise NotImplementedError(
            "the hydrostatic terrain-following core supports analytic "
            "maps= column mappings (sigma coordinates, e.g. "
            "zp = z * H(x, y)); this grid carries an embedding chart= "
            f"on {mapping.chart_coords} (a curvilinear / spherical "
            "coordinate system), which the hydrostatic model does not "
            "model yet (hydrostatic plan §7). Assemble on a maps= "
            "terrain grid, or a flat / stretched-only grid")
    entry = mapping.column_corrections.get(vertical)
    if entry is None or entry[1] != vertical:
        available = sorted(
            {base for _, base in mapping.column_corrections.values()})
        raise NotImplementedError(
            f"the hydrostatic vertical axis {vertical!r} is not the "
            "base of a single-base analytic column: the terrain "
            "Jacobian the vertical integral carries is defined only "
            "for a map p = M(z, params) whose one base coordinate is "
            f"the vertical axis. This grid's analytic columns have "
            f"base coordinate(s) {available}; pass "
            "vertical=<that axis> to hy.HydrostaticCore, or assemble "
            "on a flat / stretched-only grid")
    return entry


def jacobian_name(column: tuple[str, str]) -> str:
    """Return the column-Jacobian metric name ``d<mapped>_d<base>``."""
    mapped, base = column
    return f"d{mapped}_d{base}"
