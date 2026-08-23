"""Tests for ``CoordinateMapping.positions``: the map value at a space's nodes.

Prefix-mirrored shard of ``test_coordinate_mapping.py`` (the AGENTS
oversized-module rule); self-contained builders.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.constant import ConstantSpace

N = 8
TWO_PI = 2.0 * jnp.pi


def depth(x):
    """Smooth periodic water depth H(x)."""
    return 1.0 + 0.2 * jnp.sin(x)


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, float(TWO_PI)), name="x")


@pytest.fixture
def ms():
    return IntervalMesh(N, (0.0, 1.0), periodic=False, name="sigma")


@pytest.fixture
def mapping():
    return CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": depth})


@pytest.fixture
def grid(mx, ms, mapping):
    return Grid((mx, ms), mapping=mapping)


def nodes(grid, space, name):
    """Broadcast-ready node coordinates of ``name`` on ``space``."""
    return np.asarray(grid.evaluation_nodes(space, name).data)


def expected_z(grid, space):
    """Return the analytic ``sigma * H(x)`` at the nodes of ``space``."""
    x = nodes(grid, space, "x")
    return nodes(grid, space, "sigma") * np.asarray(depth(jnp.asarray(x)))


# ================================================================
#  Declaration surface
# ================================================================
def test_mapped_names_and_coords(mapping):
    assert mapping.mapped_names == ("z",)
    # the base coordinate first, then the parameter's coordinates
    assert mapping.mapped_coords == {"z": ("sigma", "x")}


def test_chart_maps_no_physical_coordinate():
    mapping = CoordinateMapping(
        chart={"X": lambda u, v: (jnp.cos(u), jnp.sin(u), v)})
    assert mapping.mapped_names == ()
    assert mapping.mapped_coords == {}


# ================================================================
#  The primal at the nodes
# ================================================================
def test_positions_evaluate_the_map_at_the_centres(grid, mx, ms):
    space = mx.center * ms.center
    z = grid.mapping.positions(space, "z")
    assert z.function_space.bare is space.bare
    assert z.name == "z"
    assert np.allclose(np.asarray(z.data), expected_z(grid, space))


def test_positions_follow_the_staggering(grid, mx, ms):
    space = mx.right * ms.outer
    z = grid.mapping.positions(space, "z")
    assert z.shape == (N, N + 1)
    # the parameter is one field: materialized at the centres and
    # interpolated onto the faces (the H the metric rows see), not
    # re-evaluated there
    h_faces = grid.create_field(mx.center, init=depth).to(mx.right)
    expected = (nodes(grid, space, "sigma")
                * np.asarray(h_faces.data).reshape(N, 1))
    assert np.allclose(np.asarray(z.data), expected)
    assert not np.allclose(np.asarray(z.data), expected_z(grid, space))


def test_positions_tag_uninvolved_factors_constant(mx, ms):
    mapping = CoordinateMapping(maps={"z": lambda sigma: sigma**2})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center
    z = grid.mapping.positions(space, "z")
    assert isinstance(z.function_space.factor("x"), ConstantSpace)
    assert np.allclose(np.asarray(z.data),
                       nodes(grid, space, "sigma") ** 2)


def test_positions_with_explicit_parameter_fields(grid, mx, ms):
    space = mx.center * ms.center
    h = grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)
    z = grid.mapping.positions(space, "z", params={"H": h})
    assert np.allclose(np.asarray(z.data),
                       2.0 * nodes(grid, space, "sigma"))


def test_positions_pick_the_parameters_out_of_a_state(grid, mx, ms):
    space = mx.center * ms.center
    h = grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)
    state = VectorField({"b": grid.create_field(space), "H": h})
    by_state = grid.mapping.positions(space, "z", params=state)
    by_name = grid.mapping.positions(space, "z", params={"H": h})
    assert np.array_equal(np.asarray(by_state.data),
                          np.asarray(by_name.data))
    # a state without the parameter keeps the static default
    bare = VectorField({"b": grid.create_field(space)})
    assert np.allclose(
        np.asarray(grid.mapping.positions(space, "z", params=bare).data),
        expected_z(grid, space))


def test_metric_reads_a_state_the_same_way(grid, mx, ms):
    space = mx.center * ms.center
    h = grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)
    state = VectorField({"H": h})
    by_state = grid.mapping.metric(space, "dz_dsigma", params=state)
    by_name = grid.mapping.metric(space, "dz_dsigma", params={"H": h})
    assert np.array_equal(np.asarray(by_state.data),
                          np.asarray(by_name.data))
    assert np.allclose(np.asarray(by_state.data), 2.0)


# ================================================================
#  Taught errors
# ================================================================
def test_unknown_mapped_name(grid, mx, ms):
    with pytest.raises(ValueError, match="unknown mapped coordinate 'w'"):
        grid.mapping.positions(mx.center * ms.center, "w")


def test_space_must_resolve_every_coordinate_of_the_map(grid, mx, ms):
    with pytest.raises(ValueError, match="does not resolve"):
        grid.mapping.positions(ms.center, "z")
    with pytest.raises(ValueError, match="constant along"):
        grid.mapping.positions(mx.constant * ms.center, "z")


def test_unknown_parameter_name_is_refused(grid, mx, ms):
    h = grid.create_field(mx.center, init=lambda x: 2.0 + 0.0 * x)
    with pytest.raises(ValueError, match="unknown parameters"):
        grid.mapping.positions(mx.center * ms.center, "z",
                               params={"G": h})


# ================================================================
#  Traced: jit and autodiff through a parameter field
# ================================================================
def test_positions_differentiate_through_a_parameter_field(grid, mx, ms):
    space = mx.center * ms.center

    def total(h):
        field = grid.create_field(mx.center, data=h)
        return grid.mapping.positions(
            space, "z", params={"H": field}).data.sum()

    h0 = jnp.full((N,), 1.5)
    gradient = jax.grad(total)(h0)
    assert np.all(np.isfinite(np.asarray(gradient)))
    # d/dH_i sum_ij sigma_j H_i = sum_j sigma_j, for every i
    assert np.allclose(np.asarray(gradient),
                       nodes(grid, space, "sigma").sum())
    assert np.allclose(jax.jit(total)(h0), total(h0))
