"""Distributed scan carries do not exchange discarded diagnostic halos."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body, _seal_state_ghosts
from fridom.spatial.decomposition.halo import HaloSpec


@pytest.mark.multi_device
def test_scan_state_uses_component_keys_and_preserves_true_values():
    mesh = fr.spatial.meshes.IntervalMesh(32, (0., 1.), name="x")
    grid = fr.spatial.Grid(
        (mesh,), device_ids=tuple(range(jax.device_count())))
    grid.negotiate(state_spaces=(mesh.center,), halo=HaloSpec({"x": 3}))
    field = grid.create_field(mesh.center, init=lambda x: jnp.sin(5 * x))
    vector = fr.spatial.VectorField({"carried": field, "diagnosed": field})
    got = _seal_state_ghosts(vector, ("diagnosed",))
    assert got["diagnosed"].halo_valid == HaloSpec({"x": 0})
    assert got["carried"].halo_valid == grid.decomposition.halo
    for name in vector.component_names:
        np.testing.assert_array_equal(got[name].data, vector[name].data)
    # Repeating normalization is a treedef fixed point, and an ordinary
    # consumer can still repair a reset diagnostic's ghosts correctly.
    again = _seal_state_ghosts(got, ("diagnosed",))
    assert (jax.tree_util.tree_structure(again)
            == jax.tree_util.tree_structure(got))
    np.testing.assert_array_equal(grid.sync(got["diagnosed"]).storage,
                                  grid.sync(vector["diagnosed"]).storage)


@pytest.mark.multi_device
def test_diagnostic_carry_run_gradient_matches_finite_difference():
    mesh = fr.spatial.meshes.IntervalMesh
    model = hy.Model(
        grid=fr.spatial.Grid((
            mesh(16, (0., 1.), name="x"),
            mesh(8, (0., 1.), name="y"),
            mesh(4, (0., 1.), periodic=False, name="z")),
            device_ids=tuple(range(jax.device_count()))),
        core=hy.Core(gravity=1.3),
        buoyancy=hy.ConstantStratification(n2=1.),
        advection=fr.model.modules.WENOAdvection(order=5),
        free_surface=hy.SplitExplicitFreeSurface(substeps=6),
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-4, order=2))
    model.set_fields(
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
                   for name in ("u", "v", "b", "w", "p_hyd"))

    loss = jax.jit(loss)
    got = jax.jit(jax.grad(loss))(.7)
    expected = (loss(.70001) - loss(.69999)) / 2e-5
    assert np.isfinite(got)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-10)
