"""
The lat-lon sphere convenience grid.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``. Builds the two
``IntervalMesh`` factors (periodic or sector longitude, bounded latitude)
under the orthogonal sphere chart and delegates to the base
``fridom.spatial.Grid`` — the fast assemble for a spherical grid.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fridom.spatial.charts import lonlat_sphere
from fridom.spatial.grid import Grid as _Grid
from fridom.spatial.meshes.interval import IntervalMesh

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.immersed_domain import ImmersedDomain

_HALF_PI = math.pi / 2
_TWO_PI = 2 * math.pi


class Grid(_Grid):

    r"""
    Lat-lon sphere convenience grid.

    Description
    -----------
    A thin convenience over the base :class:`fridom.spatial.Grid`: it
    builds a longitude and a latitude ``IntervalMesh`` and attaches the
    orthogonal sphere chart (``fr.spatial.charts.lonlat_sphere``), so the
    9-line hand-assembly collapses to one call. Adds no new methods or
    properties; the grid behaves exactly as one built by hand.

    Longitude is periodic over :math:`[0, 2\pi)` by default. Passing
    ``lon_extent`` instead bounds it (closed east/west walls) — a
    longitude sector. Latitude is always bounded; ``lat_extent`` is
    **required** (there is no safe default: the poles are metric-singular,
    :math:`\sqrt{g} = a\cos\phi \to 0`), and an asymmetric band such as
    ``(0, lat_max)`` gives a single-hemisphere domain.

    The prognostic velocities on the resulting grid are the contravariant
    chart components; see ``sw.modules.DynamicalCore``.

    Parameters
    ----------
    shape : tuple[int, int]
        Cell counts ``(nlon, nlat)`` for longitude and latitude.
    radius : float, optional
        The sphere radius :math:`a` (default: 1.0).
    lat_extent : tuple[float, float]
        The latitude band ``(lat_min, lat_max)`` in radians. Required.
        Must lie strictly inside ``(-pi/2, pi/2)`` (the poles are
        singular).
    lon_extent : tuple[float, float] | None, optional
        The longitude band ``(lon_min, lon_max)`` in radians. ``None``
        (the default) makes longitude periodic over ``(0, 2*pi)``;
        a value bounds longitude (closed zonal walls) — a sector.
    dispatch : object | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).
    immersed : ImmersedDomain | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).
    device_ids : tuple[int, ...] | None, optional
        Forwarded to :class:`fridom.spatial.Grid` (default: None).

    Raises
    ------
    ValueError
        If ``lat_extent`` reaches or crosses a pole (``|lat| >= pi/2``).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        radius: float = 1.0,
        *,
        lat_extent: tuple[float, float],
        lon_extent: tuple[float, float] | None = None,
        dispatch: object | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build the lon/lat meshes + sphere chart; delegate to Grid."""
        nlon, nlat = shape
        lat_min, lat_max = lat_extent
        if lat_min <= -_HALF_PI or lat_max >= _HALF_PI:
            raise ValueError(
                f"lat_extent={lat_extent} reaches a geographic pole "
                "(|lat| >= pi/2), where the sphere metric is singular "
                "(sqrt_g = a*cos(lat) -> 0). Cap the band away from "
                "the poles, e.g. lat_extent=(-1.4, 1.4) (~±80°).")
        if lon_extent is None:
            mlon = IntervalMesh(nlon, (0.0, _TWO_PI), periodic=True,
                                name="lon")
        else:
            mlon = IntervalMesh(nlon, lon_extent, periodic=False,
                                name="lon")
        mlat = IntervalMesh(nlat, lat_extent, periodic=False,
                            name="lat")
        super().__init__(
            (mlon, mlat), dispatch=dispatch,
            mapping=lonlat_sphere(radius), immersed=immersed,
            device_ids=device_ids)
