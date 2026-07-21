r"""End-to-end reverse-mode autodiff through a nonhydro2 run.

``jax.grad`` of a quadratic loss w.r.t. ``friction.nu`` through a short
advection-on run — the **masked CG pressure solve in the loop** — is
finite and matches a central finite difference, and forward-mode
``jax.jvp`` agrees with reverse mode (both AD modes stay alive; no
``custom_vjp`` in the step path). The scan-based CG differentiates
transparently at the default iteration budget (see
``design/research/jax_grad_run_investigation.md``). Pattern and recipe:
``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 0.02
TWO_PI = 2.0 * np.pi


def make_grid():
    """Return a tiny periodic 8^3 grid."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z")))


def advecting_model(nu=1e-2):
    """Return an 8^3 advection-on nonhydro2 model with friction.

    Advection is on and ``pressure_iterations`` is the default (the CG
    projection runs every step), so the gradient crosses the pressure
    solve. A sheared velocity keeps advection and friction active.
    """
    model = nh.Model(
        grid=make_grid(),
        core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=True,
        modules_extra=(fr.model.closures.HarmonicFriction(nu=nu),))
    ax = (np.arange(N) + 0.5) * (TWO_PI / N)
    x, _y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    model.set_fields(b=0.01 * np.cos(z), u=0.2 * np.sin(z),
                     v=0.2 * np.cos(x))
    return model


def nu_leaf(model):
    """Return the harmonic-friction ``nu`` leaf on the carry."""
    return next(m.nu for m in model._carry.modules
                if type(m).__name__ == "HarmonicFriction")


def nu_loss(model, n_steps=8):
    """Quadratic loss in ``friction.nu`` via the pure step kernel."""
    record = model._artifacts.record
    stepper = model._stepper
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    leaf = nu_leaf(model)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, n_steps, carry, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64)


# ================================================================
#  Reverse mode through the pressure solve: finite + FD-matched
# ================================================================
def test_grad_wrt_friction_nu_through_pressure_solve_matches_fd():
    """Check d loss / d nu across the masked CG solve: finite, FD-ok."""
    model = advecting_model()
    loss, nu0 = nu_loss(model)

    grad = float(jax.grad(loss)(nu0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0

    eps = 1e-4
    h = eps * float(nu0)
    fd = (float(loss(nu0 + h)) - float(loss(nu0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Forward mode agrees with reverse mode (both AD modes alive)
# ================================================================
def test_forward_mode_jvp_agrees_with_reverse_mode():
    """jax.jvp (forward) matches jax.grad (reverse) — no custom_vjp."""
    model = advecting_model()
    loss, nu0 = nu_loss(model)

    _, jvp = jax.jvp(loss, (nu0,),
                     (jnp.asarray(1.0, dtype=jnp.float64),))
    jvp = float(jvp)
    assert np.isfinite(jvp)
    np.testing.assert_allclose(jvp, float(jax.grad(loss)(nu0)),
                               rtol=1e-6)
