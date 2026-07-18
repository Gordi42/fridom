"""Autodiff regression tests for the boundary operators (§5).

Slice and scatter are mutual native VJPs — no ``custom_vjp`` anywhere
(step-path policy). ``jax.grad`` of a scalar loss through trace ->
scale -> scatter-add (and through embed) is finite and central-FD
matched to rtol 1e-4.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.verbs import scatter_add
from fridom.spatial.spaces.trace import Side

N = 4
EPS = 1e-4


@pytest.fixture
def grid():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 1.0), name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    return Grid((mx, my, mz))


@pytest.fixture
def space(grid):
    mx, my, mz = grid.factors
    return mx.center * my.center * mz.center


@pytest.fixture
def data0():
    return jnp.asarray(
        np.random.default_rng(0).normal(size=(N, N, N)))


def _central_fd(fn, x):
    return (fn(x + EPS) - fn(x - EPS)) / (2.0 * EPS)


def test_grad_trace_scale_scatter_matches_fd(grid, space, data0):
    def loss(c):
        f = grid.create_field(space, data=data0)
        corr = f.trace("z", Side.HIGH) * c
        tend = scatter_add(grid.create_field(space), corr)
        return jnp.sum(tend.data**2)

    grad = float(jax.grad(loss)(2.0))
    assert np.isfinite(grad)
    assert np.allclose(grad, float(_central_fd(loss, 2.0)), rtol=1e-4)


def test_grad_embed_matches_fd(grid, space, data0):
    def loss(c):
        f = grid.create_field(space, data=data0)
        sparse = (f.trace("z", Side.LOW) * c).embed("z")
        return jnp.sum(sparse.data**2)

    grad = float(jax.grad(loss)(1.5))
    assert np.isfinite(grad)
    assert np.allclose(grad, float(_central_fd(loss, 1.5)), rtol=1e-4)


def test_grad_wrt_input_field_is_finite(grid, space, data0):
    # slice <-> scatter are mutual VJPs: the reverse pass stays finite
    def loss(d):
        f = grid.create_field(space, data=d)
        tend = scatter_add(grid.create_field(space),
                           f.trace("z", Side.HIGH) * 3.0)
        return jnp.sum(tend.data**2)

    grad = jax.grad(loss)(data0)
    assert bool(np.all(np.isfinite(np.asarray(grad))))
    # only the boundary row carries gradient (scatter is a pure row op)
    g = np.asarray(grad)
    assert np.allclose(g[:, :, :-1], 0.0)
    assert bool(np.any(g[:, :, -1] != 0.0))
