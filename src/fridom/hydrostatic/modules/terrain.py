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

#: the smallest immersed quadrature ``order`` a terrain column admits.
#: Below it (``None`` / ``1``) the collocation staircase samples the
#: indicator at the *computational* cell centre, which the sigma column
#: maps to the wrong *physical* height — a silent wrong-geometry mask on
#: a chart. Composing terrain + immersed therefore requires the genuine
#: Jacobian-weighted chart fractions (MI-D1); the collocation staircase
#: is a taught error (the nonhydro2 ``_MIN_CHART_ORDER`` precedent).
_MIN_CHART_ORDER = 2


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


def require_chart_immersed_order(
    grid: Grid, column: tuple[str, str] | None,
) -> None:
    r"""Refuse a collocation-order mask on a terrain (sigma) column (M5).

    Description
    -----------
    The shared terrain + immersed composition guard (MI-D6, stage M5).
    Off a terrain grid (``column is None``) or an unmasked terrain grid
    (no immersed domain) this is a no-op — the byte-identical
    single-descriptor paths. On a **terrain + immersed** grid it requires
    the immersed domain to carry genuine per-cell Jacobian-weighted
    quadrature fractions (``ImmersedDomain(order=q)`` with
    ``q >= 2``, :data:`_MIN_CHART_ORDER`): the collocation staircase
    (``order=None`` / ``1``) samples the indicator at the *computational*
    cell centre, which the sigma column maps to the wrong *physical*
    height, so a collocation mask on a chart is a silent wrong-geometry
    mask — a taught error, never a silent unmapped mask (the nonhydro2
    ``ComposedPressureSolver`` precedent). Every hydrostatic module that
    discovers a terrain column (``HydrostaticCore``,
    ``ConstantStratification``, the free-surface variants) calls this at
    bind so the composition is refused (or admitted) uniformly.

    Parameters
    ----------
    grid : Grid
        The bound grid (carries the immersed descriptor, if any).
    column : tuple[str, str] | None
        The discovered terrain column (``None`` off a mapped grid).

    Raises
    ------
    NotImplementedError
        For a terrain + immersed grid whose immersed domain uses the
        collocation staircase (``order`` below :data:`_MIN_CHART_ORDER`).
    """
    if column is None:
        return
    immersed = getattr(grid, "immersed", None)
    if immersed is None:
        return
    order = getattr(immersed, "order", None)
    if order is None or order < _MIN_CHART_ORDER:
        raise NotImplementedError(
            "a terrain-following (sigma) column on top of an immersed "
            "(cut-cell) domain needs genuine per-cell quadrature "
            "fractions (ImmersedDomain(order=q) with q >= 2): the "
            "collocation staircase (order=None/1) samples the indicator "
            "at the computational cell centre, which the sigma column "
            "maps to the wrong physical height, so the wet-region mask "
            "would be wrong-geometry on a chart (mapped + immersed "
            "composition plan, stage M5). Pass order>=2, or use a "
            "terrain grid without an immersed mask / a flat immersed "
            "grid")
