r"""BuoyancyTracer: a bare buoyancy tracer, no background stratification.

The formulation registers ``b`` and contributes the buoyancy force
alone: no restoring term is assembled (the module is the ``N^2 = 0``
physics without paying the restoring's field arithmetic), no
``stratification.n2`` / ``stratification.froude`` is provided, and the
module is scaling-neutral. Step parity against
``ConstantStratification(n2=0)`` pins the physics; the autodiff
regression (AGENTS.md differentiability policy) runs ``jax.grad``
through the public ``Model.propagator`` w.r.t. the initial ``b``.

Self-contained per the shard convention: the small builders are
duplicated rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.buoyancy_tracer import BuoyancyTracer
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 0.02
STEPS = 8
TWO_PI = 2.0 * np.pi


def make_grid():
    """Return a tiny periodic 8^3 grid."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z")))


def make_model(buoyancy):
    """Assemble an advecting 8^3 model with the given buoyancy module."""
    return nh.Model(
        advection=nh.CenteredAdvection(),
        grid=make_grid(),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=buoyancy,
        time_stepper=AdamBashforth(DT, order=3))


def seed_fields(model, *, with_b=True):
    """Seed a sheared, divergence-free velocity (and optionally b)."""
    ax = (np.arange(N) + 0.5) * (TWO_PI / N)
    x, _y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    fields = {"u": 0.2 * np.sin(z), "v": 0.2 * np.cos(x),
              "w": 0.1 * np.sin(x)}
    if with_b:
        fields["b"] = 0.01 * np.cos(z)
    model.set_fields(**fields)


def state_sq(final):
    """Sum of squares of every final-state field (a smooth loss)."""
    return sum(jnp.sum(f.data ** 2) for f in final.state)


# ================================================================
#  Declarations and provides
# ================================================================
def test_declares_b_and_provides_no_stratification_scalar():
    module = BuoyancyTracer()
    assert [d.name for d in module.field_declarations] == ["b"]
    # scaling-neutral: no variant pin, no mechanism, no parameter
    assert getattr(module, "scaling_variant", None) is None
    assert getattr(module, "scaling_mechanism", None) is None
    model = make_model(BuoyancyTracer())
    assert "b" in model.state.component_names
    assert "stratification.n2" not in model.parameters
    assert "stratification.froude" not in model.parameters


def test_fv_family_puts_b_on_the_average_family():
    decl = BuoyancyTracer(family="fv").field_declarations[0]
    assert decl.space.family == "fv"


# ================================================================
#  Physics: no restoring; parity with the N^2 = 0 stratification
# ================================================================
def test_no_restoring_b_stays_zero_where_n2_would_move_it():
    # from (b=0, w!=0) one step leaves b exactly zero (no -N^2 w),
    # while the constant-stratification twin moves it
    tracer = make_model(BuoyancyTracer())
    seed_fields(tracer, with_b=False)
    tracer.advance(1)
    assert np.all(np.asarray(tracer.state["b"].data) == 0.0)

    strat = make_model(nh.ConstantStratification(n2=1.0))
    seed_fields(strat, with_b=False)
    strat.advance(1)
    assert np.any(np.asarray(strat.state["b"].data) != 0.0)


def test_step_parity_with_zero_n2_stratification():
    # BuoyancyTracer is ConstantStratification(n2=0) minus the
    # assembled restoring term: a short run matches to roundoff
    tracer = make_model(BuoyancyTracer())
    zero_n2 = make_model(nh.ConstantStratification(n2=0.0))
    for model in (tracer, zero_n2):
        seed_fields(model)
        model.advance(3)
    for name in ("u", "v", "w", "b"):
        np.testing.assert_allclose(
            np.asarray(tracer.state[name].data),
            np.asarray(zero_n2.state[name].data),
            rtol=1e-13, atol=1e-15, err_msg=name)


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
def test_grad_wrt_initial_b_matches_fd_directionally():
    """The IC gradient's projection onto a direction matches a FD."""
    model = make_model(BuoyancyTracer())
    seed_fields(model)
    run = model.propagator(wrt=("b",), steps=STEPS)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return state_sq(run((field,)))

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b0.shape),
                            dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
