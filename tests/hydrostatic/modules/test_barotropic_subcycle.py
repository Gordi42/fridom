"""Blocked substeps agree with the field stencil, including shard edges."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.modules.barotropic_subcycle import (
    periodic_subcycle,
    supports_blocking,
)
from fridom.hydrostatic.modules.free_surface import _sm2005_weights
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.operators.finite_difference import FiniteDifference


def _model(*, nx=32, wall=False, family="nodal"):
    mesh = fr.spatial.meshes.IntervalMesh
    return hy.Model(
        grid=fr.spatial.Grid((
            mesh(nx, (0.0, 1.0), periodic=not wall, name="x"),
            mesh(12, (0.0, 1.0), periodic=True, name="y"),
            mesh(4, (0.0, 1.0), periodic=False, name="z")),
            device_ids=tuple(device.id for device in jax.devices())),
        core=hy.Core(gravity=1.3, family=family),
        free_surface=hy.SplitExplicitFreeSurface(substeps=30),
        time_stepper=AdamBashforth(1e-4, order=2))


def _fields(model):
    rng = np.random.default_rng(51)
    return tuple(model.state[name].with_data(
        jnp.asarray(rng.normal(size=model.state[name].shape)))
        for name in ("ps", "U", "V", "U", "V"))


def _reference(fields, dtau, csqr, weights):
    p, u, v, fu, fv = fields
    ap, au, av = (field * 0.0 for field in (p, u, v))
    for weight in weights:
        p = p - dtau * csqr * (u.diff("x") + v.diff("y"))
        u = u - dtau * p.diff("x").retag(u) + dtau * fu
        v = v - dtau * p.diff("y").retag(v) + dtau * fv
        ap, au, av = ap + weight * p, au + weight * u, av + weight * v
    return ap, au, av


@pytest.mark.parametrize("substeps", [5, 16, 30, 31])
@pytest.mark.parametrize("nx", [32, 84, 128])
def test_blocked_matches_unblocked_random_fields(substeps, nx):
    model = _model(nx=nx)
    fields = _fields(model)
    assert supports_blocking(*fields[:3], ("x", "y"))
    weights = _sm2005_weights(substeps, 2, 4, 0.18927)
    actual = jax.jit(lambda f: periodic_subcycle(
        f, 1e-4, 1.3, weights, ("x", "y")))(fields)
    expected = jax.jit(lambda f: _reference(f, 1e-4, 1.3, weights))(fields)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got.data, want.data, rtol=1e-12, atol=1e-12)


def test_walls_and_alternate_stencils_fall_back(monkeypatch):
    averages = _fields(_model(family="fv"))
    assert not supports_blocking(*averages[:3], ("x", "y"))
    wall = _fields(_model(wall=True))
    assert not supports_blocking(*wall[:3], ("x", "y"))
    model = _model()
    fields = _fields(model)
    assert not supports_blocking(*fields[:3], ("y", "x"))
    original = model.grid.dispatch.resolve

    def alternate(kind, factor, *args, **kwargs):
        return (FiniteDifference(order=4) if kind == "diff" else
                original(kind, factor, *args, **kwargs))

    monkeypatch.setattr(model.grid.dispatch, "resolve", alternate)
    assert not supports_blocking(*fields[:3], ("x", "y"))


def test_blocked_kernel_gradient_matches_finite_difference():
    fields = _fields(_model(nx=16))
    weights = _sm2005_weights(6, 2, 4, 0.18927)

    def loss(amplitude):
        changed = (fields[0] * amplitude, *fields[1:])
        result = periodic_subcycle(changed, 1e-4, 1.3, weights, ("x", "y"))
        return sum(jnp.mean(field.data**2) for field in result)

    loss = jax.jit(loss)
    gradient = jax.jit(jax.grad(loss))(0.7)
    difference = (loss(0.70001) - loss(0.69999)) / 2e-5
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, difference, rtol=1e-4, atol=1e-8)


def test_periodic_model_run_gradient_matches_finite_difference():
    model = _model(nx=16)
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


def test_degenerate_and_nonconstant_columns_fall_back():
    fields = _fields(_model(nx=1))
    assert not supports_blocking(*fields[:3], ("x", "y"))
    model = _model()
    assert not supports_blocking(model.state["u"], model.state["u"],
                                 model.state["v"], ("x", "y"))


@pytest.mark.multi_device
def test_other_and_inconsistent_layouts_fall_back():
    model = _model()
    p, u, v, *_ = _fields(model)
    layout = next(layout for layout in model.grid.decomposition.layouts
                  if "y" in dict(layout.device_axes))
    other = tuple(field.reshard(layout) for field in (p, u, v))
    assert not supports_blocking(p, other[1], v, ("x", "y"))
    assert not supports_blocking(*other, ("x", "y"))


@pytest.mark.parametrize("weights", [(0.0,) * 5, (1.0,),
                                    (0.1, 0.0, 0.3, 0.0, 0.0),
                                    (0.0, 0.0, 0.0, 1.0),
                                    tuple([1.0 / 32] * 32)])
def test_full_width_and_zero_weight_patterns(weights):
    fields = _fields(_model(nx=128))
    assert fields[0].grid.decomposition.device_count == jax.device_count()
    actual = jax.jit(lambda f: periodic_subcycle(
        f, 1e-4, 1.3, weights, ("x", "y")))(fields)
    expected = jax.jit(lambda f: _reference(f, 1e-4, 1.3, weights))(fields)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got.data, want.data, rtol=1e-12, atol=1e-12)


def test_trimmed_kernel_vjp_matches_full_stencil():
    fields = _fields(_model(nx=84))
    weights = _sm2005_weights(30, 2, 4, 0.18927)
    assert len(weights) == 30
    assert weights[20] > 0
    assert not any(weights[21:])

    def loss(amplitude, kernel):
        changed = (fields[0] * amplitude, *fields[1:])
        result = kernel(changed, 1e-4, 1.3, weights)
        return sum(jnp.mean(field.data**2) for field in result)

    def blocked(f, dt, cs, w):
        return periodic_subcycle(f, dt, cs, w, ("x", "y"))

    actual = jax.jit(jax.grad(lambda a: loss(a, blocked)))(0.7)
    expected = jax.jit(jax.grad(lambda a: loss(a, _reference)))(0.7)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
