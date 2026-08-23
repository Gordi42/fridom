"""Tests for ``nodes_dataset`` (``grid.nodes`` / ``field.nodes``).

Prefix-mirrored shard of ``test_export.py`` (the AGENTS
oversized-module rule); self-contained builders.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from fridom.spatial.charts import lonlat_sphere
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.export import nodes_dataset
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def my():
    return IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my), device_ids=(0,))


def nodes(grid, space, name):
    """Return the 1D coordinate vector of one factor."""
    return np.asarray(
        grid.evaluation_nodes(space, name=name).data).reshape(-1)


def terrain(mx):
    """Build a terrain column ``zp = z H(x)`` over a periodic x."""
    mz = IntervalMesh(4, (-1.0, 0.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x)})
    return Grid((mx, mz), mapping=mapping, device_ids=(0,)), mz


# ================================================================
#  The plain-name skeleton
# ================================================================
def test_nodes_dataset_is_the_plain_name_skeleton(grid, mx, my):
    space = mx.center * my.center
    ds = grid.nodes(space)
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) == {"x", "y"}
    assert not ds.data_vars
    assert np.array_equal(ds["x"].values, nodes(grid, space, "x"))
    assert np.array_equal(ds["y"].values, nodes(grid, space, "y"))
    assert "c_grid_axis_shift" not in ds["x"].attrs


def test_staggered_space_carries_face_positions_under_plain_names(
        grid, mx, my):
    space = mx.right * my.outer
    ds = grid.nodes(space)
    assert set(ds.dims) == {"x", "y"}
    assert ds.sizes["y"] == 7
    assert ds["x"].attrs["c_grid_axis_shift"] == 0.5
    assert ds["y"].attrs["c_grid_axis_shift"] == 0.5
    assert np.array_equal(ds["x"].values, nodes(grid, space, "x"))
    assert np.array_equal(ds["y"].values, nodes(grid, space, "y"))


def test_constant_factor_is_squeezed(grid, mx, my):
    assert set(grid.nodes(mx.center * my.constant).dims) == {"x"}


def test_a_lone_factor_is_its_column(grid, my):
    ds = grid.nodes(my.outer)
    assert set(ds.dims) == {"y"}
    assert np.array_equal(ds["y"].values, nodes(grid, my.outer, "y"))


def test_field_nodes_is_the_grid_call(grid, mx, my):
    space = mx.center * my.right
    field = grid.create_field(space)
    xr.testing.assert_identical(field.nodes(), grid.nodes(space))
    xr.testing.assert_identical(nodes_dataset(grid, space),
                                grid.nodes(space))


def test_stretched_mesh_nodes_follow_its_map(mx):
    def column(s):
        return -1.0 + jnp.tanh(2.0 * s) / jnp.tanh(2.0)

    mz = MappedIntervalMesh(5, (-1.0, 0.0), column, name="z")
    grid = Grid((mx, mz), device_ids=(0,))
    ds = grid.nodes(mx.center * mz.center)
    expected = np.asarray(column((np.arange(5) + 0.5) / 5))
    assert np.allclose(ds["z"].values, expected)
    # the spacing of the column is one diff away
    assert ds["z"].diff("z").shape == (4,)


def test_chart_coordinates_carry_their_units():
    lon = IntervalMesh(8, (0.0, 2.0 * np.pi), name="lon")
    lat = IntervalMesh(4, (-1.0, 1.0), periodic=False, name="lat")
    grid = Grid((lon, lat), mapping=lonlat_sphere(), device_ids=(0,))
    ds = grid.nodes(lon.center * lat.center)
    assert ds["lon"].attrs["units"] == "rad"
    assert ds["lat"].attrs["units"] == "rad"
    # a chart maps no physical coordinate
    assert not ds.data_vars


# ================================================================
#  Mapped physical coordinates
# ================================================================
def test_mapped_coordinate_is_a_data_variable(mx):
    grid, mz = terrain(mx)
    space = mx.center * mz.outer
    ds = grid.nodes(space)
    assert ds["zp"].dims == ("x", "z")
    x = nodes(grid, space, "x")
    z = nodes(grid, space, "z")
    expected = z[None, :] * (1.0 + 0.5 * np.sin(2 * np.pi * x))[:, None]
    assert np.allclose(ds["zp"].values, expected)


def test_mapped_coordinate_under_params(mx):
    grid, mz = terrain(mx)
    space = mx.center * mz.center
    two = grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)
    by_name = grid.nodes(space, params={"H": two})
    state = VectorField({"H": two, "b": grid.create_field(space)})
    by_state = grid.nodes(space, params=state)
    z = nodes(grid, space, "z")
    assert np.allclose(by_name["zp"].values, 2.0 * z[None, :])
    xr.testing.assert_identical(by_name, by_state)
    # the reference geometry is the static default
    ref = grid.nodes(space)
    assert not np.allclose(ref["zp"].values, by_name["zp"].values)


def test_profile_space_on_a_mapped_grid_has_no_physical_coordinate(mx):
    grid, mz = terrain(mx)
    ds = grid.nodes(mz.center)
    assert set(ds.dims) == {"z"}
    assert "zp" not in ds


# ================================================================
#  Immersed domains
# ================================================================
def test_immersed_wet_mask(mx, my):
    domain = ImmersedDomain(lambda x, y: (x + 0.0 * y) < 0.5)
    grid = Grid((mx, my), immersed=domain, device_ids=(0,))
    cells = mx.center * my.center
    ds = grid.nodes(cells)
    assert ds["wet"].dtype == bool
    assert ds["wet"].dims == ("x", "y")
    assert np.array_equal(ds["wet"].values,
                          np.asarray(domain.mask(cells).data))
    # a face space carries the face mask (the slip rule), not the cells'
    faces = mx.right * my.center
    assert np.array_equal(grid.nodes(faces)["wet"].values,
                          np.asarray(domain.mask(faces).data))
    # no mask on a space that does not resolve every coordinate
    assert "wet" not in grid.nodes(my.center)


# ================================================================
#  The xarray idioms the feature is for
# ================================================================
def test_plotting_the_nodes_over_a_field(grid, mx, my):
    plt.switch_backend("Agg")
    cells = mx.center * my.center
    field = grid.create_field(cells, init=lambda x, y: x + y)
    fig, ax = plt.subplots()
    field.xr.plot(x="x", y="y", ax=ax)
    grid.nodes(cells).plot.scatter(x="x", y="y", ax=ax)
    grid.nodes(mx.right * my.center).plot.scatter(
        x="x", y="y", ax=ax, marker="|")
    plt.close(fig)


def test_stack_gives_the_list_of_points(grid, mx, my):
    ds = grid.nodes(mx.center * my.center)
    points = ds.stack(node=("x", "y"))
    assert points.sizes["node"] == 8 * 6
