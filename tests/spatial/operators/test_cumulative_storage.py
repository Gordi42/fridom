"""Transverse ghost columns survive an axis-local cumulative integral."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.cumulative import CumulativeIntegral


@pytest.mark.multi_device
@pytest.mark.parametrize("nx", [32, 33])
@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("target", ["face", "center"])
@pytest.mark.parametrize("family", ["nodal", "fv"])
@pytest.mark.parametrize("stretched", [False, True])
def test_transverse_halos_match_independent_true_column_oracle(
        nx, direction, target, family, stretched):
    mx = IntervalMesh(nx, (0., 1.), name="x")
    mz = (MappedIntervalMesh(
        8, (0., 1.), lambda z: z + .1 * jnp.sin(jnp.pi * z),
        periodic=False, name="z") if stretched else
        IntervalMesh(8, (0., 1.), periodic=False, name="z"))
    grid = Grid((mx, mz), device_ids=tuple(range(jax.device_count())))
    space = (mx.cell_avg * mz.cell_avg if family == "fv" else
             mx.center * mz.center)
    grid.negotiate(state_spaces=(space,), halo=HaloSpec({"x": 3, "z": 2}))
    values = np.random.default_rng(47).normal(size=space.shape)
    field = grid.sync(grid.create_field(space, data=jnp.asarray(values)))
    op = CumulativeIntegral(direction=direction, target=target)["z"]
    output = op(field)
    assert output.halo_valid.interval("x") == field.halo_valid.interval("x")
    assert output.halo_valid.interval("z") == (0, 0)

    # Independent NumPy prefix sums over true columns; re-pad and fill only
    # the oracle. Compare all transverse ghost columns, not just interiors.
    increments = values * np.asarray(grid.measure(space, name="z").data)
    if direction == "up":
        expected = np.concatenate((np.zeros((nx, 1)),
                                   np.cumsum(increments, axis=1)), axis=1)
    else:
        expected = np.concatenate((
            np.cumsum(increments[:, ::-1], axis=1)[:, ::-1],
            np.zeros((nx, 1))), axis=1)
    if target == "center":
        expected = .5 * (expected[:, :-1] + expected[:, 1:])
    np.testing.assert_allclose(output.data, expected, rtol=2e-14, atol=2e-14)
    oracle = grid.sync(output.with_data(jnp.asarray(expected)))
    d = grid.decomposition
    geometry = d._geometry(output.function_space, output.function_space.layout)
    width = geometry[1][4]
    n = expected.shape[1]
    np.testing.assert_allclose(output.storage[:, width:width + n],
                               oracle.storage[:, width:width + n],
                               rtol=2e-14, atol=2e-14)


@pytest.mark.multi_device
def test_partial_transverse_validity_is_not_inflated():
    mx = IntervalMesh(16, (0., 1.), name="x")
    mz = IntervalMesh(8, (0., 1.), periodic=False, name="z")
    grid = Grid((mx, mz), device_ids=tuple(range(jax.device_count())))
    space = mx.center * mz.center
    grid.negotiate(state_spaces=(space,), halo=HaloSpec({"x": 3, "z": 2}))
    field = grid.sync(grid.create_field(space, init=lambda x, z: x + z))
    partial = type(field)(grid, field.function_space, field.storage,
                          field.metadata,
                          halo_valid=HaloSpec({"x": (1, 2), "z": 0}))
    got = CumulativeIntegral()["z"](partial)
    assert got.halo_valid == HaloSpec({"x": (1, 2), "z": 0})


@pytest.mark.multi_device
def test_first_axis_integral_with_transverse_decomposition():
    mz = IntervalMesh(8, (0., 1.), periodic=False, name="z")
    mx = IntervalMesh(32, (0., 1.), name="x")
    grid = Grid((mz, mx), device_ids=tuple(range(jax.device_count())))
    space = mz.center * mx.center
    grid.negotiate(state_spaces=(space,), halo=HaloSpec({"x": 3, "z": 2}))
    field = grid.sync(grid.create_field(space, init=lambda z, x: z + x))
    output = CumulativeIntegral()["z"](field)
    values = np.asarray(field.data)
    expected = np.concatenate((np.zeros((1, 32)),
                               np.cumsum(values / 8, axis=0)), axis=0)
    np.testing.assert_allclose(output.data, expected, rtol=2e-14, atol=2e-14)
    assert output.halo_valid.interval("x") == field.halo_valid.interval("x")


@pytest.mark.multi_device
def test_jacobian_weighted_distributed_integral_retains_original_path():
    mx = IntervalMesh(32, (0., 1.), name="x")
    mz = IntervalMesh(8, (0., 1.), periodic=False, name="z")
    grid = Grid((mx, mz), device_ids=tuple(range(jax.device_count())),
                mapping=CoordinateMapping(
                    maps={"height": lambda z, depth: z * depth},
                    params={"depth": lambda x: 1. + .2 * jnp.sin(6 * x)}))
    space = mx.center * mz.center
    field = grid.create_field(space, init=lambda x, z: x + z)
    output = CumulativeIntegral(jacobian=("height",))["z"](field)
    x = np.asarray(grid.evaluation_nodes(space, "x").data)
    increments = np.asarray(field.data) * (1. + .2 * np.sin(6 * x)) / 8
    expected = np.concatenate((np.zeros((32, 1)),
                               np.cumsum(increments, axis=1)), axis=1)
    np.testing.assert_allclose(output.data, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.multi_device
def test_hydrostatic_run_gradient_through_retained_ghost_columns():
    mesh = IntervalMesh
    model = hy.Model(
        grid=Grid((mesh(16, (0., 1.), name="x"),
                   mesh(8, (0., 1.), name="y"),
                   mesh(4, (0., 1.), periodic=False, name="z")),
                  device_ids=tuple(range(jax.device_count()))),
        core=hy.Core(gravity=1.3),
        buoyancy=hy.ConstantStratification(n2=1.),
        advection=fr.model.modules.WENOAdvection(order=5),
        free_surface=hy.SplitExplicitFreeSurface(substeps=6),
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-4, order=2))
    model.set_fields(
        u=lambda x, y, z: .1 * jnp.sin(2 * jnp.pi * x) + 0 * y + 0 * z,
        b=lambda x, y, z: .01 * jnp.cos(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y) * jnp.cos(jnp.pi * z))
    carry = model._carry
    leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (index,) = [i for i, item in enumerate(leaves) if item is leaf]

    def loss(amplitude):
        changed = list(leaves)
        changed[index] = amplitude * leaf
        out = _chunk_body(model._artifacts.record, 3,
                          jax.tree_util.tree_unflatten(treedef, changed),
                          model._stepper)
        return sum(jnp.mean(out.state[name].data**2)
                   for name in ("u", "v", "b"))

    loss = jax.jit(loss)
    got = jax.jit(jax.grad(loss))(.7)
    expected = (loss(.70001) - loss(.69999)) / 2e-5
    assert np.isfinite(got)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-10)
