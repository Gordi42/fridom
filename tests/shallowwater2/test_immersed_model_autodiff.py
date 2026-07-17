r"""Reverse-mode autodiff through an immersed (cut-cell) shallowwater2 run.

The differentiability policy (AGENTS.md) applies to the masked
shallow-water step path: ``jax.grad`` of a quadratic loss through a
short masked Sadourny + gravity run is finite and matches a central
finite difference to ``rtol`` 1e-4. The differentiation variable is an
**initial field** (closures are a taught error on immersed grids). Its
data path crosses every immersed guard on the step: the safe-denominator
PV division ``_potential_vorticity`` (``sadourny.py`` — the corner
``zeta / p_full`` whose masked ``0/0`` VJP would poison every cotangent,
routed through the guard *before* the boolean wet mask) and the
double-``where`` divergence scale ``scale_divergence``
(``immersed_weighting.py`` — the ``(1/(theta V)) div`` of the linear
gravity core ``_gravity_immersed``, exact zero on a dry ``theta = 0``
cell). Genuine partial cells (a smooth analytic side wall,
``min_fraction = 0``) so the fraction weighting is a true partial, not a
``{0, 1}`` staircase. Recipe: ``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.model import _chunk_body
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _slope(x, y):  # noqa: ARG001
    """Return a smooth analytic side wall (genuine x-partial cut cells)."""
    return jnp.clip(1.4 - 0.35 * x, 0.0, 1.0)


def immersed_model(*, dt=0.01):
    """Return a tiny immersed shallowwater2 model with genuine partials.

    Advection on (masked Sadourny) and the linear gravity core
    (``_gravity_immersed``) both run, so the gradient crosses the
    guarded PV division and the fraction-weighted divergence scale.
    """
    grid = Grid(
        (IM(12, (0.0, 6.0), periodic=False, name="x"),
         IM(12, (0.0, 6.0), periodic=True, name="y")),
        immersed=ImmersedDomain(_slope, order=2, min_fraction=0.0))
    model = sw.Model(
        grid=grid, csqr=0.8, rossby_number=0.3,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))
    rng = np.random.default_rng(2)
    mask = np.asarray(
        grid.immersed.mask(model.state["p"].function_space).data)
    model.set_fields(
        p=0.1 * rng.standard_normal(model.state["p"].data.shape) * mask,
        u=0.05 * rng.standard_normal(model.state["u"].data.shape),
        v=0.05 * rng.standard_normal(model.state["v"].data.shape))
    return model


def leaf_loss(model, leaf, n_steps, *, on_stepper=False):
    """Quadratic loss splicing ``leaf`` into the carry (or stepper)."""
    record = model._artifacts.record
    tree = model._stepper if on_stepper else model._carry
    other = model._carry if on_stepper else model._stepper
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        if on_stepper:
            final = _chunk_body(record, n_steps, other, spliced)
        else:
            final = _chunk_body(record, n_steps, spliced, other)
        state = final.state
        return (jnp.sum(state["p"].data ** 2)
                + jnp.sum(state["u"].data ** 2)
                + jnp.sum(state["v"].data ** 2))

    return loss


# ================================================================
#  Genuine partial cells exist (the fraction is a true partial)
# ================================================================
def test_model_has_genuine_partial_cells():
    """The smooth wall carves cells strictly between 0 and 1."""
    model = immersed_model()
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["p"].function_space).data)
    assert ((theta > 1e-6) & (theta < 1.0 - 1e-6)).sum() > 0


# ================================================================
#  grad w.r.t. an initial field through the masked Sadourny + gravity
# ================================================================
def test_grad_wrt_initial_pressure_is_finite_and_matches_fd():
    """Grad through the masked run w.r.t. the IC: finite, FD-matched."""
    model = immersed_model()
    p_leaf = model._carry.state["p"].storage
    loss = leaf_loss(model, p_leaf, n_steps=10)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    # a masked PV 0/0 in a dry cell would NaN every entry with a data path
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(3)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  grad w.r.t. the stepper dt (the loop-invariant scalar leaf)
# ================================================================
def test_grad_wrt_stepper_dt_is_finite_and_matches_fd():
    """Grad w.r.t. dt through the masked run: finite scalar, FD-matched."""
    model = immersed_model()
    dt0 = model._stepper.dt
    loss = leaf_loss(model, dt0, n_steps=10, on_stepper=True)

    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-4 * float(dt0)
    fd = (float(loss(dt0 + h)) - float(loss(dt0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
