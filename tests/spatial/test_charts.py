"""Tests for fridom.spatial.charts."""
import numpy as np
import pytest

from fridom.spatial.charts import lonlat_sphere
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
LAT_MAX = float(np.deg2rad(80.0))


def _sphere_grid(radius=1.0):
    mlon = IntervalMesh(16, (0.0, TWO_PI), name="lon")
    mlat = IntervalMesh(8, (-LAT_MAX, LAT_MAX), periodic=False,
                        name="lat")
    return Grid((mlon, mlat), mapping=lonlat_sphere(radius))


def test_lonlat_sphere_returns_orthogonal_mapping():
    mapping = lonlat_sphere()
    assert isinstance(mapping, CoordinateMapping)
    assert mapping.orthogonal is True


def test_lonlat_sphere_binds_a_lon_lat_chart():
    grid = _sphere_grid()
    assert grid.chart_coords == ("lon", "lat")


@pytest.mark.parametrize("radius", [1.0, 2.5])
def test_lonlat_sphere_metric_is_the_analytic_sphere(radius):
    grid = _sphere_grid(radius)
    mlon, mlat = grid.factors
    space = (mlon.center * mlat.center).bare
    g_latlat = np.asarray(grid.metric(space, "g_latlat").data)
    g_lonlat = np.asarray(grid.metric(space, "g_lonlat").data)
    g_lonlon = np.asarray(grid.metric(space, "g_lonlon").data)
    lat = np.asarray(
        grid.create_field(space, init=lambda lon, lat: lat + 0.0 * lon)
        .data)
    # diagonal, and matching the closed-form lat-lon metric
    assert np.allclose(g_latlat, radius**2)
    assert np.abs(g_lonlat).max() < 1e-6
    assert np.allclose(g_lonlon, radius**2 * np.cos(lat) ** 2,
                       atol=1e-6)
