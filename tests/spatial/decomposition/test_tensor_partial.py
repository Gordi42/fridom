"""Partial periodic halo repairs retain valid layers and exact interiors."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.bc import BC
from fridom.spatial.decomposition import tensor
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import TensorDecomposition
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.spaces.nodal import NodeSet


def _case(kind="uniform", axis="x"):
    nx = 33 if kind == "uneven" else 32
    mx = (MappedIntervalMesh(
        nx, (0., 1.), lambda x: x + .01*jnp.sin(2*jnp.pi*x),
        periodic=True, name="x") if kind == "mapped" else IntervalMesh(
            nx, (0., 1.), periodic=kind != "bounded", name="x"))
    my = IntervalMesh(32, (0., 1.), name="y")
    fx = (mx.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
          if kind == "bounded" else mx.center)
    space = fx * my.center
    layout = Layout({axis: "devices"})
    decomp = TensorDecomposition((mx, my), ("x", "y"),
                                 HaloSpec({"x": 3, "y": 3}), (layout,),
                                 device_ids=tuple(range(jax.device_count())))
    raw = jnp.asarray(np.random.default_rng(927).normal(size=space.shape))
    full = decomp.sync(decomp.pad(raw, space, layout), space, layout=layout)
    return decomp, space, layout, full


def _corrupt(decomp, space, layout, full, axis, left, right):
    index = space.names.index(axis)
    _, n, factor, shards, width, _, _ = decomp._geometry(space, layout)[index]
    cells = decomp._cells_per_shard(factor, shards)
    spec = decomp.sharding(space, layout).spec

    def damage(block):
        rank = jax.lax.axis_index("devices")
        count = jnp.where(rank == shards - 1, n - (shards - 1)*cells, cells)
        for missing, start in ((width-left, 0),
                               (width-right, width+count+right)):
            if missing:
                shape = list(block.shape)
                shape[index] = missing
                block = jax.lax.dynamic_update_slice_in_dim(
                    block, jnp.full(tuple(shape), 73.), start, index)
        return block

    return jax.shard_map(damage, mesh=decomp.device_mesh,
                         in_specs=spec, out_specs=spec)(full)


@pytest.mark.multi_device
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("left", [0, 1, 2, 3])
@pytest.mark.parametrize("right", [0, 1, 2, 3])
def test_every_missing_band_pair_matches_full_sync(axis, left, right):
    decomp, space, layout, full = _case(axis=axis)
    broken = _corrupt(decomp, space, layout, full, axis, left, right)
    valid = HaloSpec({axis: (left, right)})
    got = jax.jit(lambda data: decomp.sync(data, space, layout=layout,
                                         valid=valid))(broken)
    np.testing.assert_array_equal(got, full)


@pytest.mark.multi_device
@pytest.mark.parametrize("kind", ["uneven", "bounded", "mapped"])
def test_unsupported_geometry_keeps_full_exchange(kind):
    decomp, space, layout, full = _case(kind)
    broken = _corrupt(decomp, space, layout, full, "x", 1, 2)
    valid = HaloSpec({"x": (1, 2)})
    got = jax.jit(lambda data: decomp.sync(
        data, space, layout=layout, valid=valid))(broken)
    np.testing.assert_array_equal(got, full)


@pytest.mark.multi_device
def test_partial_halo_reverse_mode_matches_finite_difference():
    decomp, space, layout, full = _case()
    broken = _corrupt(decomp, space, layout, full, "x", 1, 3)

    def loss(amplitude):
        fixed = decomp.sync(amplitude * broken, space, layout=layout,
                            valid=HaloSpec({"x": (1, 3)}))
        return jnp.mean(fixed**2)

    loss = jax.jit(loss)
    got = jax.jit(jax.grad(loss))(.7)
    want = (loss(.70001)-loss(.69999))/2e-5
    assert np.isfinite(got)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-8)
def test_periodic_weno_run_gradient_matches_finite_difference():
    mesh = fr.spatial.meshes.IntervalMesh
    model = hy.Model(
        grid=fr.spatial.Grid((
            mesh(16, (0., 1.), name="x"),
            mesh(8, (0., 1.), name="y"),
            mesh(4, (0., 1.), periodic=False, name="z"))),
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
        return sum(jnp.sum(out.state[name].data**2)
                   for name in ("b", "u", "v", "ps"))

    loss = jax.jit(loss)
    gradient = jax.jit(jax.grad(loss))(.7)
    difference = (loss(.70001) - loss(.69999)) / 2e-5
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, difference, rtol=1e-4, atol=1e-8)


@pytest.mark.multi_device
def test_halo_wider_than_local_interior_retains_full_exchange(monkeypatch):
    mx = IntervalMesh(8, (0., 1.), name="x")
    layout = Layout({"x": "devices"})
    decomp = TensorDecomposition(
        (mx,), ("x",), HaloSpec({"x": 3}), (layout,),
        device_ids=tuple(range(jax.device_count())))
    space = mx.center
    padded = decomp.pad(jnp.arange(8.), space, layout)
    expected = decomp.sync(padded, space, layout=layout)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("oversized halo must keep the original exchange")

    monkeypatch.setattr(tensor, "_exchange_periodic_bands", forbidden)
    actual = decomp.sync(padded, space, layout=layout,
                         valid=HaloSpec({"x": (1, 2)}))
    np.testing.assert_array_equal(actual, expected)
