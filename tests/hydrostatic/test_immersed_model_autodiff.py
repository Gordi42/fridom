r"""Reverse-mode autodiff through an immersed hydrostatic run.

The differentiability policy (AGENTS.md) applies to the masked
hydrostatic step path: ``jax.grad`` of a quadratic loss through a short
immersed **implicit free-surface** run is finite and matches a central
finite difference to ``rtol`` 1e-4. The differentiation variable is an
**initial field** (closures are a taught error on immersed grids). Its
data path crosses the two hydrostatic immersed hazards: the guarded
``alpha_z`` division in the continuity ``w``-diagnosis
(``core.py``, ``_diagnose_w`` / ``_masked_w_faces`` — double-``where``
so a closed ``alpha_z = 0`` face gives exact zero ``w``) and the
wet-column barotropic PCG of the implicit free surface
(``free_surface.py``: ``_guarded_inverse`` / ``_wet_depth_mean`` and the
fixed-iteration ``ConjugateGradient`` with the wet-column-masked
spectral preconditioner and the V-orthogonal wet-mean projection). A NaN
gradient would signal a masked ``0/0`` whose forward value is sealed but
whose VJP is singular. Genuine partial cells (a smooth sloping side wall,
``min_fraction = 0.1``). Recipe: ``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _sidewall(x, y, z):  # noqa: ARG001
    """Return a smooth sloping side wall ``B(y)`` (genuine partial cells)."""
    b = 0.35 + 0.1 * jnp.sin(2 * jnp.pi * y)
    return jnp.clip((x - b) / (1.0 / 8) + 0.5, 0.0, 1.0)


def immersed_model(*, dt=0.002):
    """Return a tiny immersed hydrostatic model, implicit free surface.

    The wet-column barotropic PCG runs every step
    (``pressure_iterations = 20``) and advection drives the masked
    continuity ``w``-diagnosis, so the gradient crosses the guarded
    ``alpha_z`` division and the wet-column solve.
    """
    grid = Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"),
         IM(4, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_sidewall, order=4, min_fraction=0.1))
    model = hy.Model(
        grid=grid, dt=dt, csqr=1.0,
        free_surface=hy.ImplicitFreeSurface(pressure_iterations=20),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=0.0),
        advection=True)
    rng = np.random.default_rng(11)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
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
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss


# ================================================================
#  Genuine partial cells exist (the fraction is a true partial)
# ================================================================
def test_model_has_genuine_partial_cells():
    """The smooth side wall carves cells strictly between 0 and 1."""
    model = immersed_model()
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["b"].function_space).data)
    assert ((theta > 1e-6) & (theta < 1.0 - 1e-6)).sum() > 0


# ================================================================
#  grad w.r.t. an initial field through the wet-column PCG + w-diagnosis
# ================================================================
def test_grad_wrt_initial_buoyancy_is_finite_and_matches_fd():
    """Grad through the immersed free surface w.r.t. the IC: finite, FD."""
    model = immersed_model()
    b_leaf = model._carry.state["b"].storage
    loss = leaf_loss(model, b_leaf, n_steps=6)

    grad = np.asarray(jax.grad(loss)(b_leaf))
    # a masked alpha_z 0/0 would NaN every entry with a data path
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(5)
    direction = jnp.asarray(rng.standard_normal(b_leaf.shape),
                            dtype=b_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b_leaf + eps * direction))
          - float(loss(b_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  grad w.r.t. the stepper dt (the loop-invariant scalar leaf)
# ================================================================
def test_grad_wrt_stepper_dt_is_finite_and_matches_fd():
    """Grad w.r.t. dt through the masked run: finite scalar, FD-matched."""
    model = immersed_model()
    dt0 = model._stepper.dt
    loss = leaf_loss(model, dt0, n_steps=6, on_stepper=True)

    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-4 * float(dt0)
    fd = (float(loss(dt0 + h)) - float(loss(dt0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
