"""
The Cartesian convenience grid.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``. Builds uniform
``IntervalMesh`` factors from ``shape``/``extent``/``periodic`` and
delegates to the base ``fridom.spatial.Grid`` — the fast assemble for a
uniform tensor grid.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.grid import Grid as _Grid
from fridom.spatial.meshes.interval import IntervalMesh

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.coordinate_mapping import CoordinateMapping
    from fridom.spatial.immersed_domain import ImmersedDomain

#: Default coordinate names for grids up to three dimensions.
_DEFAULT_NAMES = ("x", "y", "z")


class Grid(_Grid):

    """
    Cartesian convenience grid: uniform ``IntervalMesh`` factors.

    Description
    -----------
    A thin convenience over the base :class:`fridom.spatial.Grid`: it
    builds one uniform ``IntervalMesh`` per axis from ``shape`` /
    ``extent`` / ``periodic`` and delegates. Adds no new methods or
    properties; the two constructors stay split so neither signature
    silently accepts the other's kwargs.

    Parameters
    ----------
    shape : tuple[int, ...]
        Cell count per axis.
    extent : tuple[tuple[float, float], ...]
        The physical ``(min, max)`` interval per axis.
    periodic : bool | tuple[bool, ...], optional
        Whether each axis is periodic; a single bool broadcasts to all
        axes (default: True).
    names : tuple[str, ...] | None, optional
        Coordinate names per axis; ``None`` defaults to
        ``("x", "y", "z")[:ndim]`` and is required for ``ndim > 3``
        (default: None).
    dispatch : object | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).
    mapping : CoordinateMapping | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).
    immersed : ImmersedDomain | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).
    device_ids : tuple[int, ...] | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).

    Raises
    ------
    ValueError
        If ``extent``, ``periodic``, or ``names`` disagree with the
        number of axes in ``shape``, or ``names`` is omitted for
        ``ndim > 3``.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        extent: tuple[tuple[float, float], ...],
        periodic: bool | tuple[bool, ...] = True,
        names: tuple[str, ...] | None = None,
        *,
        dispatch: object | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build uniform IntervalMesh factors and assemble the grid."""
        ndim = len(shape)
        if len(extent) != ndim:
            raise ValueError(
                f"extent has {len(extent)} intervals but shape has "
                f"{ndim} axes")
        periodics = (
            (periodic,) * ndim if isinstance(periodic, bool)
            else tuple(periodic))
        if len(periodics) != ndim:
            raise ValueError(
                f"periodic has {len(periodics)} flags but shape has "
                f"{ndim} axes")
        if names is None:
            if ndim > len(_DEFAULT_NAMES):
                raise ValueError(
                    f"names is required for ndim > {len(_DEFAULT_NAMES)}"
                    f" (got ndim={ndim}); the default names cover only "
                    f"{_DEFAULT_NAMES}")
            names = _DEFAULT_NAMES[:ndim]
        elif len(names) != ndim:
            raise ValueError(
                f"names has {len(names)} entries but shape has {ndim} "
                "axes")
        meshes = tuple(
            IntervalMesh(n, ext, periodic=p, name=name)
            for n, ext, p, name in zip(
                shape, extent, periodics, names, strict=True))
        super().__init__(
            meshes, dispatch=dispatch, mapping=mapping,
            immersed=immersed, device_ids=device_ids)
