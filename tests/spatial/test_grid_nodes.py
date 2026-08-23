"""Tests for the mapped-name overload of ``Grid.evaluation_nodes``.

Prefix-mirrored shard of ``test_grid.py`` (the AGENTS oversized-module
rule); self-contained builders.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.charts import lonlat_sphere
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")


@pytest.fixture
def ms():
    return IntervalMesh(N, (0.0, 1.0), periodic=False, name="sigma")


@pytest.fixture
def grid(mx, ms):
    return Grid((mx, ms), mapping=CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x)}))


@pytest.fixture
def two(grid, mx):
    """Return a constant depth of two, as a parameter field."""
    return grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)


def test_a_mapped_name_routes_to_positions(grid, mx, ms):
    space = mx.center * ms.center
    via_grid = grid.evaluation_nodes(space, "z")
    direct = grid.mapping.positions(space, "z")
    assert via_grid.function_space is direct.function_space
    assert np.array_equal(np.asarray(via_grid.data),
                          np.asarray(direct.data))


def test_a_mapped_name_threads_params(grid, mx, ms, two):
    space = mx.center * ms.center
    z = grid.evaluation_nodes(space, "z", params={"H": two})
    sigma = np.asarray(grid.evaluation_nodes(space, "sigma").data)
    assert np.allclose(np.asarray(z.data), 2.0 * sigma)


def test_field_evaluation_nodes_threads_params(grid, mx, ms, two):
    space = mx.center * ms.center
    field = grid.create_field(space)
    z = field.evaluation_nodes("z", params={"H": two})
    ref = grid.evaluation_nodes(space, "z", params={"H": two})
    assert np.array_equal(np.asarray(z.data), np.asarray(ref.data))
    # the plain-coordinate forwarder is unchanged
    assert np.array_equal(
        np.asarray(field.evaluation_nodes("sigma").data),
        np.asarray(grid.evaluation_nodes(space, "sigma").data))


def test_unknown_name_names_the_coordinates_and_the_maps(grid, mx, ms):
    with pytest.raises(KeyError, match=r"no factor along 'w'.*"
                       r"resolves \('x', 'sigma'\) and the mapping maps "
                       r"\('z',\)"):
        grid.evaluation_nodes(mx.center * ms.center, "w")


def test_unknown_name_without_a_mapping(mx, ms):
    flat = Grid((mx, ms))
    with pytest.raises(KeyError, match="no factor along 'z'") as info:
        flat.evaluation_nodes(mx.center * ms.center, "z")
    assert "mapping maps" not in str(info.value)


def test_unknown_name_on_a_chart_names_no_maps():
    lon = IntervalMesh(N, (0.0, 2.0 * np.pi), name="lon")
    lat = IntervalMesh(N, (-1.0, 1.0), periodic=False, name="lat")
    sphere = Grid((lon, lat), mapping=lonlat_sphere())
    with pytest.raises(KeyError, match="no factor along 'zp'") as info:
        sphere.evaluation_nodes(lon.center * lat.center, "zp")
    assert "mapping maps" not in str(info.value)


def test_params_are_refused_on_a_grid_coordinate(grid, mx, ms, two):
    with pytest.raises(ValueError, match="params= applies to a mapped"):
        grid.evaluation_nodes(mx.center * ms.center, "sigma",
                              params={"H": two})
    with pytest.raises(ValueError, match="params= applies to a mapped"):
        grid.evaluation_nodes(ms.center, params={})
