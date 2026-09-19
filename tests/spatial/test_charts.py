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


def test_lonlat_sphere_declares_its_coordinates_as_radians():
    # the row in model.units converts to metres of arc (factor L), so
    # reading that row as a CF claim labelled radians as metres
    mapping = lonlat_sphere()
    assert mapping.coordinate_units == {"lon": "rad", "lat": "rad"}


def test_sphere_grid_forwards_the_declared_coordinate_units():
    grid = _sphere_grid()
    assert grid.coordinate_units == {"lon": "rad", "lat": "rad"}


# ================================================================
#  The torus preset (spherical-models plan, SP-D3)
# ================================================================
def _torus_grid(major=2.0, minor=0.5):
    from fridom.spatial.charts import torus  # noqa: PLC0415
    mu = IntervalMesh(16, (0.0, TWO_PI), name="tor")
    mv = IntervalMesh(12, (0.0, TWO_PI), name="pol")
    return Grid((mu, mv), mapping=torus(major, minor))


def test_torus_is_an_orthogonal_wall_free_chart():
    grid = _torus_grid()
    assert grid.mapping.orthogonal is True
    assert grid.chart_coords == ("tor", "pol")
    assert all(mesh.periodic for mesh in grid.factors)


def test_torus_metric_is_analytic_and_never_zero():
    major, minor = 2.0, 0.5
    grid = _torus_grid(major, minor)
    mu, mv = grid.factors
    space = (mu.center * mv.center).bare
    v = np.asarray(grid.evaluation_nodes(space, "pol").data)
    ring = np.broadcast_to(major + minor * np.cos(v), space.shape)
    v = np.broadcast_to(v, space.shape)
    np.testing.assert_allclose(
        np.asarray(grid.metric(space, "g_tortor").data), ring ** 2,
        rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(grid.metric(space, "g_polpol").data),
        minor ** 2 + 0.0 * ring,
        rtol=1e-13)
    sqrt_g = np.asarray(grid.metric(space, "sqrt_g").data)
    np.testing.assert_allclose(sqrt_g, minor * ring, rtol=1e-13)
    assert sqrt_g.min() > 0.0
    # the one nonzero scale-factor derivative: d h_u / d v
    np.testing.assert_allclose(
        np.asarray(grid.metric(space, "dh_tor_dpol").data),
        -minor * np.sin(v), atol=1e-13)


@pytest.mark.parametrize("radii", [(1.0, 1.0), (1.0, 2.0), (2.0, 0.0)])
def test_torus_rejects_a_degenerate_ring(radii):
    from fridom.spatial.charts import torus  # noqa: PLC0415
    with pytest.raises(ValueError, match="ring torus"):
        torus(*radii)
