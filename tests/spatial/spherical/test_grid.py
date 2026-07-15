"""Tests for fridom.spatial.spherical.grid."""
import math

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid as BaseGrid
from fridom.spatial.scalars import Variance
from fridom.spatial.spherical.grid import Grid

TWO_PI = 2.0 * math.pi
LAT_MAX = float(np.deg2rad(80.0))


@pytest.fixture
def sphere():
    return Grid((16, 8), radius=2.0, lat_extent=(-LAT_MAX, LAT_MAX))


# ================================================================
#  Construction: meshes, names, chart
# ================================================================
def test_is_a_grid_subclass(sphere):
    assert isinstance(sphere, BaseGrid)


def test_default_periodic_lon_bounded_lat(sphere):
    mlon, mlat = sphere.factors
    assert sphere.names == ("lon", "lat")
    assert mlon.periodic is True
    assert mlon.extent == (0.0, TWO_PI)
    assert mlat.periodic is False
    assert mlat.extent == (-LAT_MAX, LAT_MAX)


def test_attaches_the_orthogonal_sphere_chart(sphere):
    assert sphere.chart_coords == ("lon", "lat")


def test_radius_scales_the_metric():
    grid = Grid((16, 8), radius=3.0, lat_extent=(-LAT_MAX, LAT_MAX))
    mlon, mlat = grid.factors
    space = (mlon.center * mlat.center).bare
    g_latlat = np.asarray(grid.metric(space, "g_latlat").data)
    assert np.allclose(g_latlat, 9.0)


# ================================================================
#  Extents: hemispheres and longitude sectors
# ================================================================
def test_northern_hemisphere_lat_extent():
    grid = Grid((16, 8), lat_extent=(0.0, LAT_MAX))
    assert grid.factors[1].extent == (0.0, LAT_MAX)


def test_longitude_sector_closes_the_zonal_boundaries():
    grid = Grid((8, 8), lat_extent=(0.0, LAT_MAX),
                lon_extent=(0.0, math.pi / 2))
    mlon = grid.factors[0]
    assert mlon.periodic is False
    assert mlon.extent == (0.0, math.pi / 2)


# ================================================================
#  Index moves seed diagonal (orthogonal chart) and assemble
# ================================================================
def test_orthogonal_chart_seeds_diagonal_index_moves(sphere):
    mlon, mlat = sphere.factors
    space = (mlon.center * mlat.center).bare
    assert sphere.dispatch.resolve("raise_index", space).diagonal is True
    assert sphere.dispatch.resolve("lower_index", space).diagonal is True


def _covariant_vector(grid):
    mlon, mlat = grid.factors
    uu = grid.create_field(
        mlon.right * mlat.center,
        init=lambda lon, lat: jnp.sin(lon + lat)).with_variance(
            Variance.COVARIANT)
    vv = grid.create_field(
        mlon.center * mlat.right,
        init=lambda lon, lat: jnp.cos(lon - lat)).with_variance(
            Variance.COVARIANT)
    return VectorField({"u": uu, "v": vv})


def test_index_raise_assembles_across_the_latitude_wall(sphere):
    vec = _covariant_vector(sphere)
    raised = sphere.dispatch.resolve(
        "raise_index", vec[0].function_space.bare)(vec)
    for comp in raised:
        assert comp.function_space.variance is Variance.CONTRAVARIANT


def test_index_raise_assembles_across_a_sector_all_walls():
    grid = Grid((8, 8), lat_extent=(0.0, LAT_MAX),
                lon_extent=(0.0, math.pi / 2))
    vec = _covariant_vector(grid)
    raised = grid.dispatch.resolve(
        "raise_index", vec[0].function_space.bare)(vec)
    for comp in raised:
        assert comp.function_space.variance is Variance.CONTRAVARIANT


# ================================================================
#  Pole guard
# ================================================================
@pytest.mark.parametrize(
    "lat_extent",
    [(-math.pi / 2, LAT_MAX), (0.0, math.pi / 2), (0.0, 2.0),
     (-2.0, 0.0)])
def test_pole_touching_lat_extent_is_rejected(lat_extent):
    with pytest.raises(ValueError, match="pole"):
        Grid((8, 8), lat_extent=lat_extent)
