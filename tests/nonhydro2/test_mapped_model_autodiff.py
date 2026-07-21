r"""Reverse-mode autodiff through a mapped (terrain-following) run.

The differentiability policy (AGENTS.md, "Differentiability policy")
applies to the mapped step path: ``jax.grad`` of a quadratic loss
through a short terrain-following run is finite and matches a central
finite difference to ``rtol`` 1e-4. The interesting path is the
**flux-consistent mapped pressure projection** — divergence, the
fixed-iteration mapped PCG solve, and the velocity correction
(:class:`MappedPressureSolver`, ``_project_mapped``). The correction's
metric quotient ``F_i / J`` is the reverse hazard: the column Jacobian
``J = dm/db`` is strictly positive on every valid cell but zero-filled
in the never-valid storage padding, where the raw divide is a sealed
``inf`` the retag discards in the primal but whose VJP is singular
(``0 * inf -> NaN``). The differentiation variable is an **initial
field** (mapped grids reject closures, so ``friction.nu`` is
unavailable) whose data path crosses the projection every step. A NaN
gradient here is a bug — the masked-singularity poison the policy
targets. Both nodal and FV families run the same guarded divide, so the
regression covers both. Recipe: ``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

TWO_PI = 2.0 * np.pi
IM = IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _depth(x):
    """Return a smooth periodic water depth H(x) (40% slope)."""
    return 1.0 + 0.4 * jnp.sin(x)


def mapped_model(*, dt=0.02, family="nodal", pressure_iterations=12,
                 pressure_tolerance=1e-8):
    """Return a tiny mapped nonhydro2 model on ``zp = z * H(x)``.

    The terrain-following column couples the vertical to ``x`` through
    the mapping, so every step's projection runs the mapped PCG and its
    flux-consistent velocity correction — the ``F_i / J`` metric
    quotient whose reverse VJP the guard seals. ``family`` selects the
    nodal C-grid or the FV default (both share the mapped solver).
    """
    mx = IM(8, (0.0, TWO_PI), periodic=True, name="x")
    my = IM(8, (0.0, TWO_PI), periodic=True, name="y")
    mz = IM(8, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": _depth})
    grid = Grid((mx, my, mz), mapping=mapping)
    model = nh.Model(
        grid=grid,
        core=nh.Core(
            family=family,
            pressure_iterations=pressure_iterations,
            pressure_tolerance=pressure_tolerance),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=False)
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
#  grad w.r.t. an initial field through the mapped projection
# ================================================================
@pytest.mark.parametrize("family", ["nodal", "fv"])
def test_grad_wrt_initial_velocity_is_finite_and_matches_fd(family):
    """Grad through the mapped projection w.r.t. the IC: finite, FD.

    The reverse regression for the mapped velocity correction: a NaN
    here would mean the ``F_i / J`` metric quotient's singular divide
    (``J == 0`` in never-valid padding) leaks through the VJP. Both the
    nodal C-grid and the FV default share :class:`MappedPressureSolver`,
    so both must differentiate cleanly.
    """
    model = mapped_model(family=family)
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=4)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    # a masked 0/0 in the Jacobian padding would NaN every entry
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
#  Stretched base + multigrid: the eager coarse-grid pre-warm path
# ================================================================
def _stretch(z):
    """Monotone sigma clustering built from a jnp map (dS/dz > 0)."""
    return z + 0.15 * jnp.sin(2.0 * np.pi * z) / (2.0 * np.pi)


def stretched_multigrid_model(*, dt=0.02, pressure_iterations=12):
    """Return a tiny STRETCHED terrain model on the multigrid solve.

    The vertical is a stretched ``MappedIntervalMesh`` whose jnp map
    means its coarse level's host-validated ctor cannot run under a
    dynamic trace, combined with the ``zp = z H(x)`` terrain map. A
    stretched base rejects the spectral preconditioner (N1), so the
    multigrid V-cycle solves — taking the GM-D9 full-coarsening default
    through the eager coarse-grid pre-warm
    (:meth:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver._prewarm_hierarchy`).
    Building the model already exercises the model path: the assembly
    dry-run abstract-traces the projection, so the pre-warm must have
    warmed ``Grid.coarsened``'s memo (under ``ensure_compile_time_eval``)
    or the coarse ctor would raise a ``TracerArrayConversionError`` under
    that trace.
    """
    mx = IM(8, (0.0, TWO_PI), periodic=True, name="x")
    my = IM(8, (0.0, TWO_PI), periodic=True, name="y")
    mz = MappedIntervalMesh(8, (0.0, 1.0), _stretch, periodic=False,
                            name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": _depth})
    grid = Grid((mx, my, mz), mapping=mapping)
    model = nh.Model(
        grid=grid,
        core=nh.Core(
            family="fv",
            pressure_iterations=pressure_iterations,
            pressure_preconditioner="multigrid"),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=False)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


# reverse-mode through the FULL-coarsening multigrid projection now builds
# on >1 device: the fine->coarse transfer's ``restrict`` spells ``P^T``
# with forward primitives so the transpose of ``jnp.roll`` no longer lands
# in the forward graph where the XLA SPMD partitioner miscompiled it at one
# cell per device (transfer.py). Runs on any device count.
def test_stretched_multigrid_grad_wrt_initial_velocity_matches_fd():
    """Grad through the stretched full-coarsening multigrid projection.

    The model-path regression for the eager coarse-grid pre-warm (record
    §Residue 3): assembling the model already forces the projection's
    dry-run trace to memo-hit the coarse hierarchy, and the reverse
    gradient through a short run stays finite and matches a central FD —
    the added coarse levels contribute only static geometry, no new
    masked singularity beyond the ones the mapped/stretched guards
    already seal.
    """
    model = stretched_multigrid_model()
    u_leaf = model._carry.state["u"].storage
    loss = leaf_loss(model, u_leaf, n_steps=3)

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
#  grad w.r.t. the stepper dt (the loop-invariant scalar leaf)
# ================================================================
def test_grad_wrt_stepper_dt_is_finite_and_matches_fd():
    """Grad w.r.t. dt through the mapped run: finite scalar, FD-matched.

    ``dt`` threads the correction scaling of every substage, so its
    reverse path also crosses the guarded metric quotient.
    """
    model = mapped_model()
    dt0 = model._stepper.dt
    loss = leaf_loss(model, dt0, n_steps=4, on_stepper=True)

    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-4 * float(dt0)
    fd = (float(loss(dt0 + h)) - float(loss(dt0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
