r"""Reverse-mode autodiff through an immersed (cut-cell) nonhydro2 run.

The differentiability policy (AGENTS.md, "Differentiability policy")
applies to the immersed step path: ``jax.grad`` of a quadratic loss
through a short masked run is finite and matches a central finite
difference to ``rtol`` 1e-4. The interesting path is the **masked
fixed-iteration CG pressure solve** (``ImmersedPressureSolver``,
designed differentiable via ``lax.scan``); the differentiation variable
is an **initial field** (closures are a taught error on immersed grids,
so ``friction.nu`` is unavailable here) whose data path crosses every
immersed guard: the fraction-weighted advective flux and the guarded
divergence scale ``_immersed_scale`` (``advection.py``, double-``where``
so a dry ``theta = 0`` cell stays exact zero) and the wet-mean nullspace
projection of the CG (``immersed_pressure.py``). A NaN gradient here is a
bug (a masked ``0/0`` whose forward value is sealed but whose VJP is
singular). Genuine partial cells (smooth analytic fractions,
``min_fraction = 0.1``) so the fraction weighting is a true partial, not
a ``{0, 1}`` staircase. Recipe: ``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
IM = IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _slope(x, y, z):  # noqa: ARG001
    """Return a smooth analytic side wall (genuine x-partial cut cells)."""
    return jnp.clip(1.4 - 0.35 * x, 0.0, 1.0)


def immersed_model(*, dt=0.01, pressure_iterations=12,
                   pressure_tolerance=None):
    """Return a tiny immersed nonhydro2 model with genuine partials.

    Advection on and the masked CG runs every step, so the gradient
    crosses the immersed fraction weighting and the wet-mean-projected
    pressure solve. ``pressure_tolerance`` opts the solve into the
    masked-scan convergence break (default off, the fixed budget).
    """
    grid = Grid(
        (IM(8, (0.0, 6.0), periodic=False, name="x"),
         IM(8, (0.0, TWO_PI), periodic=True, name="y"),
         IM(6, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_slope, order=2, min_fraction=0.1))
    model = nh.Model(
        grid=grid,
        core=nh.Core(
            pressure_iterations=pressure_iterations,
            pressure_tolerance=pressure_tolerance),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=True)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def leaf_loss(model, leaf, n_steps, *, on_stepper=False):
    """Quadratic loss splicing ``leaf`` into the carry (or stepper).

    ``leaf`` is located by object identity; the loss advances
    ``n_steps`` through the pure kernel ``_chunk_body`` (the public
    ``advance`` path is not differentiable) and reduces the final state
    to ``sum(field**2)``.
    """
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
    """The smooth wall carves cells strictly between 0 and 1."""
    model = immersed_model()
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["b"].function_space).data)
    assert ((theta > 1e-6) & (theta < 1.0 - 1e-6)).sum() > 0


# ================================================================
#  grad w.r.t. an initial field through the masked CG solve
# ================================================================
def test_grad_wrt_initial_velocity_is_finite_and_matches_fd():
    """Grad through the immersed pressure solve w.r.t. the IC: finite, FD."""
    model = immersed_model()
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=6)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    # a masked 0/0 in a dry cell would NaN every entry with a data path
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(1)
    direction = jnp.asarray(rng.standard_normal(u_leaf.shape),
                            dtype=u_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u_leaf + eps * direction))
          - float(loss(u_leaf - eps * direction))) / (2.0 * eps)
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


# ================================================================
#  grad through the masked CG under the opt-in convergence tolerance
# ================================================================
def test_grad_through_tolerance_mode_masked_cg_matches_fd():
    """Grad through the tolerance-mode masked CG w.r.t. the IC: FD-ok.

    The masked-scan convergence break differentiates the *truncated*
    algorithm (``lax.scan`` + ``lax.cond``, no ``custom_vjp``), so the
    reverse gradient stays exact — the repo invariant. The tolerance
    (``1e-6``) fires here around iteration 14, inside the 20-step budget,
    so the cond-skip branch is genuinely exercised in both AD modes.
    The mapped CG consumer carries its own regression in
    ``test_mapped_model_autodiff.py`` (its former metric-singularity
    NaN is fixed — the guarded ``_divide_by_jacobian``).
    """
    model = immersed_model(pressure_iterations=20,
                           pressure_tolerance=1e-6)
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=6)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(2)
    direction = jnp.asarray(rng.standard_normal(u_leaf.shape),
                            dtype=u_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u_leaf + eps * direction))
          - float(loss(u_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Multigrid pressure preconditioner selected (B5 gate GB-4/T3)
# ================================================================
def multigrid_model(*, dt=0.01, levels=3):
    """Return the same immersed model with the multigrid preconditioner.

    Identical genuine-partial slope wall as :func:`immersed_model`, but
    the fixed-iteration CG pressure solve runs the semicoarsened
    geometric-multigrid V-cycle (``pressure_preconditioner="multigrid"``,
    multigrid pathway plan §B5). The 8x8x6 grid semicoarsens the
    horizontal x/y (8 -> 4, z kept), so the realized hierarchy is two
    levels. The differentiation therefore crosses the vertical-line
    smoother's batched Thomas solve **and** its dry-cell
    double-``jnp.where`` guard on the line bands — the exact masked
    singularity (``diag -> 1``, ``rhs -> 0`` on a dry column) the
    differentiability policy targets for the B1/B2 smoother code.
    """
    grid = Grid(
        (IM(8, (0.0, 6.0), periodic=False, name="x"),
         IM(8, (0.0, TWO_PI), periodic=True, name="y"),
         IM(6, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_slope, order=2, min_fraction=0.1))
    model = nh.Model(
        grid=grid,
        core=nh.Core(
            pressure_iterations=12,
            pressure_preconditioner="multigrid",
            multigrid_levels=levels),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=True)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def test_grad_through_multigrid_solve_is_finite_and_matches_fd():
    """Grad through the multigrid-preconditioned masked solve: finite, FD.

    The B5 differentiability regression for ``preconditioner="multigrid"``
    (T3): a NaN here would mean the line-smoother Thomas solve or its
    dry-cell double-``jnp.where`` guard is not reverse-safe on the true
    partial cells this model carries. The initial velocity field is the
    differentiation variable because its data path flows through the
    residual the smoother relaxes at every level.
    """
    model = multigrid_model()
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=6)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(1)
    direction = jnp.asarray(rng.standard_normal(u_leaf.shape),
                            dtype=u_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u_leaf + eps * direction))
          - float(loss(u_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  grad through the multigrid solve with agglomeration ON (MG-D10)
# ================================================================
def immersed_mg_model(*, agglomerate, dt=0.01, pressure_iterations=8):
    """Return a tiny immersed model whose masked CG uses agglomeration.

    The multigrid preconditioner runs with coarse-grid agglomeration
    ON. On one device the switch is inert (every coarse level is already
    Layout({})); in the multi-device suite the coarse levels replicate
    across the switch, so the restrict all-gather / prolong local-slice
    reshard is on the differentiated path. Either way a NaN gradient is
    a bug (the differentiability policy, knob ON).
    """
    grid = Grid(
        (IM(8, (0.0, 6.0), periodic=False, name="x"),
         IM(8, (0.0, TWO_PI), periodic=True, name="y"),
         IM(6, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_slope, order=2, min_fraction=0.1))
    model = nh.Model(
        grid=grid,
        core=nh.Core(
            pressure_iterations=pressure_iterations,
            pressure_preconditioner="multigrid",
            multigrid_tridiagonal_method="scan",
            multigrid_agglomerate=agglomerate),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=True)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def test_grad_with_agglomerate_on_is_finite_and_matches_fd():
    """Agglomerated multigrid solve: grad w.r.t. the IC is FD-matched."""
    model = immersed_mg_model(agglomerate=2)
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=4)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(1)
    direction = jnp.asarray(rng.standard_normal(u_leaf.shape),
                            dtype=u_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u_leaf + eps * direction))
          - float(loss(u_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
