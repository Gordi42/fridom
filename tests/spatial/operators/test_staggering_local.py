"""Distributed stencils need no exchange beyond their existing input halos."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import TensorDecomposition
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.staggering import _run_storage_kernel
from fridom.spatial.operators.stencil_kernels import apply_stencil


def _operand(nx, axis, *, periodic=True, single=False):
    meshes = (IntervalMesh(nx, (0., 1.), periodic=periodic, name="x"),
              IntervalMesh(12, (0., 1.), name="y"))
    space = meshes[0].center * meshes[1].center
    layout = Layout({axis: "devices"}) if not single else Layout({})
    decomp = TensorDecomposition(
        meshes, ("x", "y"), HaloSpec({"x": 3, "y": 3}),
        (layout,), device_ids=(0,) if single else None)
    field = SimpleNamespace(grid=SimpleNamespace(decomposition=decomp),
                            function_space=space.with_layout(layout))
    return field, decomp, space, layout


@pytest.mark.multi_device
@pytest.mark.parametrize("nx", [32, 33])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("coprofile", [False, True])
def test_local_windows_match_periodic_random_oracle(nx, axis, coprofile):
    f, decomp, space, layout = _operand(nx, axis)
    a = space.names.index(axis)
    values = jnp.asarray(np.random.default_rng(821).normal(size=space.shape))
    data = decomp.sync(decomp.pad(values, space, layout), space, layout=layout)
    # A full co-operand and a transverse broadcast profile exercise both
    # input partition cases. A sign-changing product is data-sensitive.
    factor = jnp.where(data > 0, 0.75, -0.25)
    if coprofile:
        shape = list(data.shape)
        shape[a] = 1
        factor = jnp.broadcast_to(jnp.asarray(0.75), tuple(shape))
    weights = (1 / 12, -2 / 3, 0., 2 / 3, -1 / 12)

    def kernel(block, index, other):
        return apply_stencil(block * other, index, weights)

    def run(block, other):
        return _run_storage_kernel(
            f, kernel, block, a, (other,), size=5, k0=2,
            s_out=data.shape[a], patched=False)

    compiled = jax.jit(run).lower(data, factor).compile()
    got = decomp.unpad(compiled(data, factor), space, layout)
    products = values * (
        0.75 if coprofile else jnp.where(values > 0, .75, -.25))
    want = sum(w * jnp.roll(products, 2 - k, axis=a)
               for k, w in enumerate(weights))
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)
    # This is the optimization's observable contract: already halo-filled
    # inputs suffice. No extra collective belongs in the stencil kernel.
    hlo = compiled.as_text().lower()
    assert "collective-permute(" not in hlo
    assert "all-gather(" not in hlo
    assert "all-to-all(" not in hlo


@pytest.mark.multi_device
def test_distributed_stencil_gradient_matches_finite_difference():
    f, decomp, space, layout = _operand(32, "x")
    values = jnp.asarray(np.random.default_rng(24).normal(size=space.shape))

    def loss(amplitude):
        data = decomp.sync(decomp.pad(amplitude * values, space, layout),
                           space, layout=layout)
        result = _run_storage_kernel(
            f, lambda block, axis: apply_stencil(block, axis, (-1., 1.)),
            data, 0, (), size=2, k0=0, s_out=data.shape[0], patched=False)
        return jnp.mean(decomp.unpad(result, space, layout)**2)

    loss = jax.jit(loss)
    got = jax.jit(jax.grad(loss))(.7)
    want = (loss(.70001) - loss(.69999)) / 2e-5
    assert np.isfinite(got)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-8)


@pytest.mark.multi_device
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
