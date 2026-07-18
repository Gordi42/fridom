r"""
The ``jacobian=`` weight shared by ``Integral`` and ``CumulativeIntegral``.

Description
-----------
The reductions weight a chart coordinate's increment by the metric
area/volume element (rules section 3.13). Two chart geometries feed
the same ``jacobian=`` seam, and both keep the documented convention
that ``jacobian=`` names *chart coordinates*:

- **embedding ``chart=``** (CS-D1): ``jacobian=`` names the chart's
  base coordinates (themselves mesh axes). While the operand space
  still resolves *every* named chart coordinate, the reduction of one
  of them carries the induced-metric ``sqrt_g`` once — the standard
  area element :math:`\int f\,\sqrt{g}\,du\,dv`.
- **analytic ``maps=``** (terrain-following column): ``jacobian=``
  names the *mapped physical* coordinate ``p`` (e.g. ``"zp"`` for
  ``zp = sigma * H(x)``), which is **not** a mesh axis. The reduction
  of its single base axis ``b`` carries the signed column Jacobian
  ``d<p>_d<b>`` — equal to the single-column ``sqrt_g`` restriction,
  positive on a monotone map, so the running/total integral becomes
  the physical :math:`\int f\,dz_p`.

The gate keys on the chart-coordinate -> base-axis relation
(``mapping.column_corrections`` for maps=), not on comparing the
chart name to the axis name — spelling the mapped coordinate as the
base axis (``jacobian=("sigma",)`` for ``zp``) therefore no longer
half-matches. A ``jacobian=`` name that names neither an embedding
chart coordinate nor an analytic map's physical coordinate resolves
to nothing on the grid and raises a taught error (no silent no-op).

The maps= branch additionally carries an **aliveness guard** (Hazard
1): the column Jacobian ``d<mapped>_d<base>`` varies over the map's
parameter axes, so it is well defined only while every coordinate the
column couples (the base and its parameters) is still resolved by the
operand space. Reducing a parameter axis to a ``ConstantSpace`` before
the base axis leaves the metric unresolvable, so the branch raises a
taught ``ValueError`` naming the coupled coordinates and the fix
(reduce the base axis first). The seeded ``f.integrate()`` /
``f.mean()`` verbs order the reduction so the base axis goes first and
never trip it (``scalar_field._bases_first``); only a hand-built
reduction that collapses a parameter axis in a separate call can.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.spaces.coefficient import CoefficientSpace

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.spaces.function_space import FunctionSpace


def resolves(space: FunctionSpace | object,
             names: tuple[str, ...]) -> bool:
    """
    Whether ``space`` resolves every name through a live factor.

    Description
    -----------
    The chart-reduction guard (rules 3.13): an embedding chart's
    ``sqrt_g`` enters an increment only while every named chart
    coordinate is still contributed by a non-collapsed (non-constant,
    non-trace), non-coefficient factor of the operand space. A field
    born constant, traced, or lone-factor along a chart coordinate
    carries no geometry and contracts against the plain computational
    measure.

    Parameters
    ----------
    space : SpaceLike
        The (bare) operand space.
    names : tuple[str, ...]
        The chart coordinate names.

    Returns
    -------
    bool
        True iff every name is contributed by a non-collapsed,
        non-coefficient factor of ``space``.
    """
    for name in names:
        try:
            factor = space.factor(name)
        except KeyError:
            return False
        if isinstance(factor, CoefficientSpace) or factor.collapses_axis:
            return False
    return True


def jacobian_factor(
    f: FieldLike, axis: str, jacobian: tuple[str, ...] | None,
) -> Array | None:
    r"""
    Return the Jacobian weight for ``axis``'s increment, or None.

    Description
    -----------
    Validates that every name in ``jacobian`` names a chart
    coordinate of the field's grid (an embedding ``chart=`` base
    coordinate, or an analytic ``maps=`` physical coordinate); a name
    that resolves to neither raises a taught ``ValueError`` naming the
    grid's available chart coordinates (killing the silent-no-op
    trap). Returns the metric array a caller multiplies onto the
    axis' increment, or None when the axis carries no Jacobian
    weight:

    - the ``sqrt_g`` area element, once, while the operand space
      resolves every named embedding-chart coordinate (module
      docstring), or
    - the column Jacobian ``d<mapped>_d<axis>`` when ``axis`` is the
      single base of a ``maps=`` column whose mapped physical
      coordinate is named in ``jacobian`` — guarded (Hazard 1): if a
      coordinate the column couples has already collapsed, a taught
      ``ValueError`` fires (reduce the base axis before its
      parameters) rather than an unresolvable-metric error.

    Parameters
    ----------
    f : FieldLike
        The operand field (its grid supplies the mapping/metrics).
    axis : str
        The resolved coordinate axis being reduced.
    jacobian : tuple[str, ...] | None
        The declared chart-coordinate family, or None for the plain
        computational measure (returns None without validation).

    Returns
    -------
    Array | None
        The weight array to multiply onto the axis' increment, or
        None when the axis carries no Jacobian weight.
    """
    if jacobian is None:
        return None
    grid = f.grid
    bare = f.function_space.bare
    mapping = grid.mapping
    chart = None if mapping is None else mapping.chart_coords
    columns = ({} if mapping is None
               else mapping.column_corrections)
    # base axis -> mapped physical name (single-base analytic columns)
    column_by_base = {base: mapped
                      for mapped, base in set(columns.values())}
    chart_coords = frozenset(chart or ())
    known = chart_coords | frozenset(column_by_base.values())
    for name in jacobian:
        if name not in known:
            raise ValueError(_unknown_name_message(name, jacobian,
                                                   known))
    # The two chart geometries are disjoint per grid — an embedding
    # chart and terrain maps never co-key one axis in any real
    # configuration (the mapped physical names are not mesh axes, and
    # a grid's geometry is one form), so at most one branch applies;
    # a chart, if somehow also declared, takes precedence.
    #
    # (a) embedding chart: the sqrt_g area element, once, while the
    #     operand space resolves every named chart coordinate
    if (axis in chart_coords and axis in jacobian
            and resolves(bare, tuple(
                n for n in jacobian if n in chart_coords))):
        return grid.metric(bare, "sqrt_g").data
    # (b) analytic maps= column whose base is this axis: the signed
    #     column Jacobian d<mapped>_d<axis> (the single-column sqrt_g
    #     restriction, positive on a monotone map)
    mapped = column_by_base.get(axis)
    if mapped is not None and mapped in jacobian:
        # aliveness guard (Hazard 1): d<mapped>_d<axis> varies over the
        # map's parameter axes, so if a coupled coordinate has already
        # reduced to a ConstantSpace the metric is unresolvable. The
        # seeded verbs reduce the base axis first (scalar_field
        # `_bases_first`); a hand-built reduction that collapses a
        # parameter axis first hits this taught error rather than a
        # silently broadcast wrong weight.
        coupled = tuple(coord for coord, pair in columns.items()
                        if pair == (mapped, axis))
        if not resolves(bare, coupled):
            raise ValueError(
                _collapsed_axis_message(mapped, axis, coupled))
        return grid.metric(bare, f"d{mapped}_d{axis}").data
    return None


def _collapsed_axis_message(
    mapped: str, axis: str, coupled: tuple[str, ...],
) -> str:
    """Taught error for a column Jacobian on a collapsed parameter."""
    return (
        f"reducing {axis!r} with the {mapped!r} column Jacobian "
        f"d{mapped}_d{axis} needs every coupled coordinate {list(coupled)}"
        f" still resolved by the operand space, but one has already "
        f"reduced to a ConstantSpace (or coefficient factor); the "
        f"Jacobian varies over the map's parameter axes, so reduce the "
        f"base axis {axis!r} before its parameters — the seeded "
        f"f.integrate() / f.mean() verbs do this automatically")


def _unknown_name_message(
    name: str, jacobian: tuple[str, ...], known: frozenset[str],
) -> str:
    """Taught error for a ``jacobian=`` name off the grid's charts."""
    if known:
        detail = f"available chart coordinates: {sorted(known)}"
    else:
        detail = "this grid declares no chart coordinates"
    return (
        f"jacobian={jacobian} names {name!r}, which is not a chart "
        f"coordinate of this grid; jacobian= names the chart "
        f"coordinates a reduction weights by — an embedding "
        f"chart's base coordinates (the sqrt_g area element) or an "
        f"analytic map's physical coordinate (its terrain-column "
        f"Jacobian); " + detail)
