"""Partial repair retains truthful claims, cache identity and derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.tensor import TensorDecomposition
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.operators.base import _ensure_valid
from fridom.spatial.spaces.nodal import NodeSet


def _partial():
    mesh = fr.spatial.meshes.IntervalMesh
    mx = mesh(16, (0., 1.), name="x")
    my = mesh(8, (0., 1.), name="y")
    mz = mesh(4, (0., 1.), periodic=False, name="z")
    grid = fr.spatial.Grid((mx, my, mz),
                          device_ids=tuple(d.id for d in jax.devices()))
    data = np.random.default_rng(51).normal(size=(16, 8, 4))
    space = mx.center * my.center * mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    f = grid.create_field(space, data=jnp.asarray(data))
    full = grid.sync(f)
    d = grid.decomposition
    geometry = d._geometry(f.function_space, f.function_space.layout)
    block = geometry[0][-1] // d.device_count
    xwidth, zwidth = geometry[0][4], geometry[2][4]
    arr = full.storage
    local_x = jnp.arange(arr.shape[0]) % block
    right_start = xwidth + geometry[0][1] // d.device_count
    xbad = (local_x >= right_start) & (local_x < right_start + xwidth)
    zbad = ((jnp.arange(arr.shape[2]) < zwidth)
            | (jnp.arange(arr.shape[2]) >= arr.shape[2] - zwidth))
    arr = jnp.where(xbad[:, None, None] | zbad[None, None, :], jnp.nan, arr)
    valid = HaloSpec({"x": (xwidth, 0), "y": geometry[1][4], "z": 0})
    partial = ScalarField(grid, f.function_space, arr, f.metadata,
                          halo_valid=valid)
    return partial, full


@pytest.mark.multi_device
def test_partial_cache_accumulates_axes_and_repairs_corners(monkeypatch):
    field, full = _partial()
    calls = []
    original = TensorDecomposition._exchange_axis

    def exchange(self, *args, **kwargs):
        calls.append(args[1])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TensorDecomposition, "_exchange_axis", exchange)
    old_tree = jax.tree_util.tree_structure(field)
    required_z = {"z": (1, 1)}
    local = _ensure_valid(field, required_z)
    assert calls == []
    assert local.halo_valid.interval("x") == (1, 0)
    assert local.halo_valid.covers("z", (1, 1))
    assert _ensure_valid(field, required_z) is local
    finished = _ensure_valid(field, {"x": (1, 1), "z": (1, 1)})
    assert calls == [0]
    assert _ensure_valid(field, required_z) is finished
    assert jax.tree_util.tree_structure(field) == old_tree
    assert field.halo_valid.interval("z") == (0, 0)
    np.testing.assert_array_equal(finished.storage, full.storage)


def test_empty_and_invalid_axis_selection():
    field, _ = _partial()
    empty = field.grid.sync(field, axes=())
    assert empty.halo_valid == field.halo_valid
    np.testing.assert_array_equal(empty.storage, field.storage)
    with pytest.raises(ValueError, match="axes must belong"):
        field.grid.sync(field, axes=("missing",))
    with pytest.raises(ValueError, match="axes must belong"):
        field.grid.decomposition.sync(
            field.storage, field.function_space, axes=("missing",))


def test_concrete_operand_never_caches_tracers():
    field, full = _partial()

    def loss(amplitude):
        ready = _ensure_valid(field, {"z": (1, 1)})
        return amplitude * jnp.sum(ready.data**2)

    expected = 0.7 * jnp.sum(full.data**2)
    np.testing.assert_allclose(jax.jit(loss)(0.7), expected, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(loss)(0.7), expected, rtol=1e-12)


def test_partial_repair_kernel_gradient():
    _, full = _partial()
    # Correct storage with conservative partial claims; avoid NaN inputs to
    # reverse mode while exercising the same partial-repair control path.
    field = ScalarField(full.grid, full.function_space, full.storage,
                        full.metadata,
                        halo_valid=HaloSpec({"x": (1, 0), "y": 1, "z": 0}))

    def loss(amplitude):
        result = (amplitude * field).diff("z").diff("x")
        return jnp.mean(result.data**2)

    loss = jax.jit(loss)
    gradient = jax.jit(jax.grad(loss))(0.7)
    difference = (loss(0.70001) - loss(0.69999)) / 2e-5
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, difference, rtol=1e-4, atol=1e-8)


def test_model_gradient_through_selective_repairs():
    mesh = fr.spatial.meshes.IntervalMesh
    grid = fr.spatial.Grid((mesh(16, (0., 1.), name="x"),
                            mesh(8, (0., 1.), name="y"),
                            mesh(4, (0., 1.), periodic=False, name="z")),
                           device_ids=tuple(d.id for d in jax.devices()))
    model = hy.Model(grid=grid, core=hy.Core(gravity=1.3),
                     buoyancy=hy.ConstantStratification(n2=1.),
                     free_surface=hy.SplitExplicitFreeSurface(substeps=6),
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         1e-4, order=2))
    model.set_fields(ps=lambda x, y: 0.01 * jnp.sin(2 * jnp.pi * x)
                     * jnp.cos(2 * jnp.pi * y))
    carry = model._carry
    record, stepper = model._artifacts.record, model._stepper
    leaf = carry.state["ps"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (index,) = [i for i, item in enumerate(leaves) if item is leaf]

    def loss(amplitude):
        changed = list(leaves)
        changed[index] = amplitude * leaf
        state = _chunk_body(record, 3, jax.tree_util.tree_unflatten(
            treedef, changed), stepper)
        return sum(jnp.sum(state.state[name].data**2)
                   for name in ("ps", "u", "v"))

    loss = jax.jit(loss)
    gradient = jax.jit(jax.grad(loss))(0.7)
    difference = (loss(0.70001) - loss(0.69999)) / 2e-5
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, difference, rtol=1e-4, atol=1e-8)

