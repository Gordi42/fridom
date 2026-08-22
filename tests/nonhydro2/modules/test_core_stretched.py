r"""Core: the projection on a bare **stretched** vertical (A1).

A grid whose vertical is a plain
:class:`~fridom.spatial.meshes.mapped_interval.MappedIntervalMesh` — a
boundary-layer-refined column — declares no analytic map, so it sets no
``mapping.column_corrections``. It is not a flat grid either: a
coordinate-mapped mesh carries no spectral basis, so the separable
:class:`SpectralPressureSolver` has no ``transform`` row on the
vertical factor and the projection used to die on a bare
``DispatchError``. :func:`~fridom.nonhydro2.modules.core._stretched_column`
now routes such a grid to the mapped PCG — the degenerate **identity
column** — on both families, and the auto preconditioner resolves to
the V-cycle (the spectral inverse would be rejected at construction,
N1).

A prefix-mirrored shard of ``test_core`` (AGENTS oversized-module
rule): the model-level runs here are jit-compilation-bound, so they
stay off the cheap assembly-only guards in the parent file.
Self-contained builders per that rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.errors import TermEvaluationError
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.composed import Divergence

N = 8
DT = 0.01
TWO_PI = 2.0 * np.pi


def _stretch(s):
    """Monotone clustering of [0, 1] (dS/ds in [0.85, 1.15] > 0)."""
    return s + 0.15 * jnp.sin(2 * np.pi * s) / (2 * np.pi)


def stretched_grid():
    """Periodic x/y, a bare stretched (unmapped) vertical column."""
    return Grid((
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y"),
        MappedIntervalMesh(N, (0.0, 1.0), _stretch, name="z")),
        device_ids=(0,))


def stretched_model(*, seed=0, **core_kwargs):
    """Return a tiny linear model on the bare stretched column."""
    model = nh.Model(
        grid=stretched_grid(),
        core=nh.Core(**core_kwargs),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=None)
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def max_divergence(model):
    """Return the C-grid divergence of the model's velocity trio."""
    state = model.state
    vel = VectorField({c: state[c] for c in ("u", "v", "w")})
    return float(np.abs(np.asarray(Divergence()(vel).data)).max())


# ================================================================
#  The projection runs (A1) and constrains the velocity
# ================================================================
def test_stretched_column_model_projects_divergence_free():
    """The bare stretched vertical assembles, runs and projects.

    The A1 regression: before the routing fix this raised
    ``DispatchError: no operator registered for kind 'transform' on
    CellAvg(z, ...)`` out of the flat spectral solve.
    """
    model = stretched_model()
    before = max_divergence(model)
    assert before > 1e-3  # the random IC really is divergent
    model.advance(steps=3)
    assert max_divergence(model) < 1e-8 * before
    for comp in ("u", "v", "w", "b"):
        assert np.all(np.isfinite(np.asarray(model.state[comp].data)))


@pytest.mark.parametrize("family", ["nodal", "fv"])
def test_stretched_column_assembles_on_both_families(family):
    """Both families route to the mapped solve, not the spectral one.

    The assembly dry-run abstract-traces the projection, so building
    the model is already the regression: the nodal grid used to raise
    the same ``DispatchError`` on ``Center(z, ...)``.
    """
    model = stretched_model(
        family=family, pressure_preconditioner="none")
    assert model.state["p"].data.shape == (N, N, N)


@pytest.mark.parametrize("preconditioner", ["multigrid", "none"])
def test_stretched_column_honours_an_explicit_preconditioner(
        preconditioner):
    """Both PCG preconditioners that serve a stretched column agree."""
    model = stretched_model(
        pressure_preconditioner=preconditioner,
        pressure_iterations=120, pressure_tolerance=1e-12)
    model.advance(steps=1)
    assert max_divergence(model) < 1e-9


# ================================================================
#  Differentiability (AGENTS "Differentiability policy")
# ================================================================
def test_grad_through_the_stretched_projection_matches_fd():
    """Grad w.r.t. the initial u through the identity-column solve.

    The identity column reaches the mapped correction's ``F_i / J``
    metric quotient with a constant-one ``J``, and the stretch rides
    the ``grid.measure`` divisions of the flux legs instead — neither
    may leak a singular VJP. Kept on the plain-CG preconditioner: the
    reverse path through the *multigrid* V-cycle on a stretched column
    is already covered by
    ``test_mapped_model_autodiff.test_stretched_multigrid_grad_wrt_initial_velocity_matches_fd``,
    and differentiating it here costs minutes for no new coverage.
    """
    model = stretched_model(
        pressure_preconditioner="none", pressure_iterations=8)
    u_leaf = model._carry.state["u"].storage
    record = model._artifacts.record
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is u_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 3, carry, model._stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

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
#  The same routing on an immersed (cut-cell) stretched grid
# ================================================================
def immersed_stretched_model(*, seed=0, **core_kwargs):
    """Build the stretched column again, under a cut-cell topography."""
    slope = lambda x, y, z: jnp.clip(  # noqa: E731, ARG005
        (z - 0.2 - 0.1 * jnp.sin(x)) * 6 + 0.5, 0.0, 1.0)
    grid = Grid((
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y"),
        MappedIntervalMesh(N, (0.0, 1.0), _stretch, name="z")),
        immersed=ImmersedDomain(slope, order=4), device_ids=(0,))
    model = nh.Model(
        grid=grid, core=nh.Core(**core_kwargs),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=None)
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def test_immersed_stretched_column_model_runs():
    """An immersed grid with a stretched vertical runs on auto (A1).

    The masked route serves the stretch in its own operator (plan §6),
    but its *spectral* preconditioner has no transform on the stretched
    factor — so before the auto fix this model died with the same
    ``DispatchError`` the unmasked stretched column did.
    """
    model = immersed_stretched_model()
    model.advance(steps=2)
    for comp in ("u", "v", "w", "b"):
        assert np.all(np.isfinite(np.asarray(model.state[comp].data)))


def test_immersed_stretched_column_refuses_explicit_spectral():
    """Explicit spectral on a stretched cut-cell grid is taught.

    It used to be a bare ``DispatchError`` raised deep inside the
    masked preconditioner build; the taught ``NotImplementedError``
    reaches the caller wrapped in the stage's ``TermEvaluationError``.
    """
    with pytest.raises(TermEvaluationError,
                       match="carries no spectral basis"):
        immersed_stretched_model(pressure_preconditioner="spectral")
