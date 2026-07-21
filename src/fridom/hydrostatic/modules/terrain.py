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

import jax.numpy as jnp

import fridom as fr
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.verbs import scatter_set
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.trace import Side

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField
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
            "vertical=<that axis> to hy.Core, or assemble "
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
    discovers a terrain column (``Core``,
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


# ================================================================
#  Physical vertical velocity: the slope terms on the w faces
# ================================================================
def slope_velocity_on_w(
    u: ScalarField, v: ScalarField, w_ref: ScalarField,
    column: tuple[str, str], horizontal: tuple[str, str], vertical: str,
) -> ScalarField:
    r"""Return the physical-w slope terms ``u Z_x + v Z_y`` on the w faces.

    Description
    -----------
    The half of the physical vertical velocity
    ``w_{phys} = J\omega + u\,Z_x + v\,Z_y`` that the contravariant flux
    ``J\omega`` omits — the coordinate-surface slope advection (rules
    3.8; the physical fluid follows the tilted sigma surfaces). The
    slopes ``Z_i = d<mapped>_d<axis>`` are read from ``grid.metric`` on
    the ``w`` face set (``Center`` horizontal, ``Outer`` vertical), and
    ``u`` / ``v`` are carried onto that set in two hops: the horizontal
    interpolation onto the cell centres (``.to`` the collocated cell,
    the same ``Right -> Center`` average the ``b`` cell uses) and the
    vertical ``Center -> Outer`` lift — the one-sided ``LinearInterp``
    variant that closes the bottom / surface faces by second-order
    linear extrapolation (the interior order), so the term is defined on
    the terrain face where the flux vanishes. The result rides the same
    ``w`` face set as the flux, ready to add.

    This is the exact quantity :meth:`Core._diagnose_w` adds
    to the flux to store the physical ``w`` and that ``State.chart``
    subtracts back to recover ``J\omega``; sharing one spelling makes
    the round trip cancel to machine precision (and exactly at the
    terrain face, where the flux seed is a hard zero).

    Parameters
    ----------
    u : ScalarField
        The physical zonal velocity (on its C-grid face).
    v : ScalarField
        The physical meridional velocity (on its C-grid face).
    w_ref : ScalarField
        A field on the ``w`` face set — supplies the target space and
        the ``grid.metric`` slope-lookup space (the diagnosed ``w`` in
        the core, the stored ``w`` in the chart derivation).
    column : tuple[str, str]
        The ``(mapped, base)`` terrain column (``discover_column``).
    horizontal : tuple[str, str]
        The (zonal, meridional) coordinate names naming the slope
        metrics ``d<mapped>_d<axis>``.
    vertical : str
        The vertical coordinate name (the ``Center -> Outer`` lift axis).

    Returns
    -------
    ScalarField
        ``u Z_x + v Z_y`` on the ``w`` face set (BC-free).
    """
    mapped, _base = column
    zonal, meridional = horizontal
    grid = w_ref.grid
    bare = w_ref.function_space.bare
    cell = fr.spatial.Collocated().resolve(grid)
    lift = LinearInterp(target=NodeSet.OUTER, boundary="one_sided")[vertical]
    u_w = lift(u.to(cell))
    v_w = lift(v.to(cell))
    zx = grid.metric(bare, f"d{mapped}_d{zonal}")
    zy = grid.metric(bare, f"d{mapped}_d{meridional}")
    return u_w * zx + v_w * zy


def masked_w_faces(
    immersed: object, state: VectorField, vertical: str,
    *, include_bottom: bool = False,
) -> ScalarField:
    r"""Return ``alpha_z`` on the ``w`` faces (boundary = cell fraction).

    Description
    -----------
    The min-rule face fraction on the vertical ``Outer`` faces treats
    the exterior beyond a **physical** boundary as dry, which would zero
    the surface (top) face and destroy the barotropic surface DOF
    ``w(0)`` (the column-divergence carrier under a free surface). A
    physical boundary is not an immersed dry region, so its face
    fraction is the adjacent cell fraction — the mirror-exterior
    convention. The **surface** (top) face is always overridden.

    The physical **bottom** face is overridden only when
    ``include_bottom`` (the physical-``w`` case): the contravariant flux
    seeds ``transport == 0`` at the bottom, so the *flux* division and
    the flat immersed diagnosis need no bottom override (``w`` is zero
    there irrespective of ``alpha_z``). But the stored physical ``w`` on
    a **terrain** column carries the slope terms ``u Z_x + v Z_y`` at the
    bed (nonzero over a slope), so the wet mask that gates the slope add
    must treat the physical bottom face as wet (its cell fraction),
    otherwise the ``jnp.where(wet, ...)`` would wrongly zero the bed
    slope on a face the min-rule marked exterior-dry. Interior cut faces
    keep their ``alpha_z == 0`` and stay dry either way.

    Each override rides the sanctioned boundary machinery
    (``design/plans/active/boundary_trace_plan.md`` §3) rather than raw
    ``.data`` surgery: the boundary cell fraction is a boundary trace of
    ``theta_cell`` (on the ``Center`` cells), relocated onto the vertical
    ``Outer`` face set through the sanctioned Constant bridge
    (``as_profile`` -> ``adopt``), then ``scatter_set`` overwrites the
    boundary ``Outer`` face of ``alpha_z``. The path is a pure slice +
    two retags + a row-scatter (all native VJPs, no ``.data``), so the
    masked wet-face indicator is bitwise stable. Shared by the core's
    masked ``w`` diagnosis and the ``State.chart`` flux derivation.

    Parameters
    ----------
    immersed : object
        The grid's immersed descriptor.
    state : VectorField
        The current state (supplies the ``w`` and cell spaces).
    vertical : str
        The vertical coordinate name.
    include_bottom : bool, optional
        Also override the physical bottom face with its cell fraction —
        the physical-``w`` terrain case (default: False, surface only).

    Returns
    -------
    ScalarField
        The ``alpha_z`` field with the physical boundary faces
        overridden.
    """
    alpha_z = immersed.fraction(state["w"].function_space)
    theta_cell = immersed.fraction(state["p_hyd"].function_space)
    # a physical boundary face fraction = the adjacent cell fraction:
    # trace the boundary Center cell, relocate it onto the Outer face set
    # (the sanctioned Constant bridge), and overwrite the boundary face.
    sides = [Side.HIGH, Side.LOW] if include_bottom else [Side.HIGH]
    for side in sides:
        boundary = (theta_cell.trace(vertical, side)
                    .as_profile(vertical)
                    .adopt(vertical, NodeSet.OUTER, side))
        alpha_z = scatter_set(alpha_z, boundary)
    return alpha_z


# ================================================================
#  Chart-native derivation (state.chart, ruling (d))
# ================================================================
def vertical_column(grid: Grid) -> tuple[str, str] | None:
    r"""Return the ``(mapped, base)`` vertical sigma column, or ``None``.

    Description
    -----------
    The lenient column discovery the ``State.chart`` derivation uses:
    the single-base analytic column whose base coordinate is itself
    (``column_corrections[axis][1] == axis`` — the vertical column, as
    opposed to a coupled horizontal coordinate that references it).
    ``None`` off a mapped grid or on an embedding chart (``chart_coords``
    is not ``None``), where the vertical component is already physical /
    chart-native and the chart view is the identity. Unlike
    :func:`discover_column` this never raises — a read-only diagnostic
    view degrades to the identity rather than teaching a taught error the
    assembled core already raised.

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
    r"""Return a hydrostatic state component in its chart-native form.

    Description
    -----------
    The per-package derivation hook behind ``hy.State.chart`` (ruling
    (d)). Every component is the **identity** except the vertical
    velocity ``w`` on a terrain (sigma) column, which becomes the
    **contravariant volume flux** ``J\omega = w - (u\,Z_x + v\,Z_y)`` —
    the stored physical ``w`` minus the slope terms
    (:func:`slope_velocity_on_w`, the exact spelling the core added). The
    subtraction recovers ``J\omega`` to machine precision, and exactly
    zero on the terrain face (the flux seed). On an unmapped grid, on an
    embedding chart, and for the horizontal / tracer components the
    stored physical field is already the chart-native quantity, so the
    identity is returned.

    On a **terrain + immersed** grid the flux is masked back onto the wet
    ``w`` faces (:func:`masked_w_faces`): a dry face carries no flux
    (``0``), matching the masked physical ``w`` the core stored there.

    Parameters
    ----------
    state : VectorField
        The hydrostatic state (physical components).
    name : str
        The component name to view.

    Returns
    -------
    ScalarField
        The chart-native component (the flux for terrain ``w``, else the
        stored physical field).
    """
    field = state[name]
    if name != "w":
        return field
    grid = field.grid
    column = vertical_column(grid)
    if column is None:
        return field
    _mapped, base = column
    horizontal = tuple(a for a in grid.names if a != base)
    slope = slope_velocity_on_w(
        state["u"], state["v"], field, column, horizontal, base)
    flux = field - slope.retag(field)
    immersed = getattr(grid, "immersed", None)
    if immersed is not None:
        # mask the flux back onto the wet faces the physical w carries:
        # the same both-boundary wet mask the core stored w with, so a
        # dry cut face reads 0 (matching the stored physical w there).
        az = masked_w_faces(immersed, state, base, include_bottom=True).data
        flux = flux.with_data(jnp.where(az > 0.0, flux.data, 0.0))
    return flux
