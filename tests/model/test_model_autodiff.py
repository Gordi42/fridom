r"""Reverse-mode autodiff through a model run — the canonical pattern.

The invariant
-------------
The new-stack step path is reverse-mode differentiable end to end:
``jax.grad`` of a scalar loss through a short model run, w.r.t. any bound
parameter (a closure coefficient, the stepper ``dt``, an initial field),
is finite and exact to finite-difference precision. This is a **tested
invariant, not an accident** — treat a NaN gradient as a bug (see
``design/research/jax_grad_run_investigation.md``).

The recipe (copy this)
----------------------
The public ``Model.advance`` path is *not* differentiable (eager
coefficient coercion, an AOT cache key that reads ``leaf.sharding``,
carry donation, per-chunk host syncs). Differentiate the **pure kernel**
``fridom.model.model._chunk_body(record, n_steps, carry, stepper)``
instead — a pure ``lax.scan`` over the ``ModelState`` pytree:

1. build the model, ``set_fields`` the initial condition;
2. grab ``record = model._artifacts.record``, ``carry = model._carry``,
   ``stepper = model._stepper``;
3. find the differentiation leaf by **identity** in ``tree_flatten`` of
   the carry (a coefficient / initial field) or the stepper (``dt``);
4. splice the traced input into that leaf, ``tree_unflatten``, run
   ``_chunk_body``, reduce the returned state to a pure-``jnp`` scalar;
5. ``jax.grad`` and validate against a central finite difference
   (relative step ~1e-4, ``rtol=1e-4``).

Keep it cheap: <=16^2 / 8^3 grids, <=10 steps, one test per feature.
Future step-path features copy this file's pattern into their mirrored
test (AGENTS.md, "Differentiability policy").
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.closures.diffusion import HarmonicDiffusion
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 12
L = 1.0
DT = 2e-3
KAPPA = 4e-3


class Core(Module):

    """Toy core: one velocity u, two tracers b/c, a trivial term."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.tracer("b"),
        fr.model.FieldDeclaration.tracer("c"),
    )

    @fr.model.term(advances=("u", "b", "c"), linear=True,
             transports=("u", "b", "c"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name] for name in ("u", "b", "c")}


def make_grid():
    """Return a tiny periodic (x, z) grid."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=True, name=name)
        for name in ("x", "z")))


def make_model(kappa=KAPPA, dt=DT):
    """Build the toy model: Core + harmonic tracer mixing."""
    model = Model(
        grid=make_grid(),
        modules=(Core(), HarmonicDiffusion(kappa)),
        time_stepper=AdamBashforth(dt, order=3))
    ax = (np.arange(N) + 0.5) * (L / N)
    x, z = np.meshgrid(ax, ax, indexing="ij")
    # smooth tracers so the harmonic mixing has a non-trivial data path
    model.set_fields(b=np.sin(2 * np.pi * x), c=np.sin(4 * np.pi * z))
    return model


def kernel(model):
    """Return (record, carry, stepper) for the pure step kernel."""
    return (model._artifacts.record, model._carry, model._stepper)


def state_sq(final):
    """Sum of squares of every final-state field (a smooth loss)."""
    return sum(jnp.sum(f.data ** 2) for f in final.state)


def leaf_loss(record, carry, stepper, leaf, n_steps, *, on_stepper=False):
    r"""Return a loss splicing ``leaf`` into the carry (or stepper).

    ``leaf`` is located by object identity in the flattened tree; the
    loss substitutes the differentiation variable there and reduces the
    ``n_steps`` output state to ``sum(field**2)``.
    """
    tree = stepper if on_stepper else carry
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        if on_stepper:
            final = _chunk_body(record, n_steps, carry, spliced)
        else:
            final = _chunk_body(record, n_steps, spliced, stepper)
        return state_sq(final)

    return loss


def central_fd(loss, x0, eps=1e-4):
    """Central finite difference of ``loss`` at ``x0`` (relative step)."""
    h = eps * (abs(float(x0)) if float(x0) != 0.0 else 1.0)
    return (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)


def kappa_leaf(carry):
    """Return the harmonic-mixing coefficient leaf on the carry."""
    return next(m for m in carry.modules
                if isinstance(m, HarmonicDiffusion)).kappa


# ================================================================
#  (i) grad w.r.t. a Harmonic closure coefficient
# ================================================================
def test_grad_wrt_closure_coefficient_is_finite_and_matches_fd():
    """Check d loss / d kappa is finite and matches a central FD."""
    record, carry, stepper = kernel(make_model())
    k0 = kappa_leaf(carry)
    loss = leaf_loss(record, carry, stepper, k0, n_steps=10)

    grad = float(jax.grad(loss)(k0))
    assert np.isfinite(grad)
    # kappa damps the tracers, so the gradient is genuinely non-zero
    assert abs(grad) > 0.0
    fd = central_fd(loss, k0)
    assert grad == pytest.approx(fd, rel=1e-4)


# ================================================================
#  (ii) grad w.r.t. the stepper dt
# ================================================================
def test_grad_wrt_stepper_dt_matches_fd():
    """Check d loss / d dt (the loop-invariant stepper leaf) vs FD."""
    record, carry, stepper = kernel(make_model())
    dt0 = stepper.dt
    loss = leaf_loss(record, carry, stepper, dt0, n_steps=10,
                     on_stepper=True)

    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    fd = central_fd(loss, dt0)
    assert grad == pytest.approx(fd, rel=1e-4)


# ================================================================
#  (iii) grad w.r.t. an initial field (dot-product / directional test)
# ================================================================
def test_grad_wrt_initial_field_matches_fd_directionally():
    """The IC gradient's projection onto a random direction matches FD."""
    record, carry, stepper = kernel(make_model())
    b_leaf = carry.state["b"].storage
    loss = leaf_loss(record, carry, stepper, b_leaf, n_steps=10)

    grad = np.asarray(jax.grad(loss)(b_leaf))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b_leaf.shape),
                            dtype=b_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b_leaf + eps * direction))
          - float(loss(b_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  (iv) chunk composition: grad(two 5-step) == grad(one 10-step)
# ================================================================
def test_grad_is_invariant_to_chunk_splitting():
    r"""Grad through two 5-step chunks equals one 10-step chunk.

    ``_chunk_body`` is a pure scan, so splitting a run into consecutive
    chunks is exact forward *and* reverse: the composed gradient must
    match the single-chunk gradient to machine precision.
    """
    record, carry, stepper = kernel(make_model())
    k0 = kappa_leaf(carry)
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is k0]

    def spliced(x):
        new = list(leaves)
        new[idx] = x
        return jax.tree_util.tree_unflatten(treedef, new)

    def loss_single(x):
        return state_sq(_chunk_body(record, 10, spliced(x), stepper))

    def loss_split(x):
        mid = _chunk_body(record, 5, spliced(x), stepper)
        return state_sq(_chunk_body(record, 5, mid, stepper))

    g_single = float(jax.grad(loss_single)(k0))
    g_split = float(jax.grad(loss_split)(k0))
    np.testing.assert_allclose(g_split, g_single, rtol=1e-11)
