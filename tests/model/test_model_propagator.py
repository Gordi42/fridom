r"""``Model.propagator``: the public differentiable run surface.

The load-bearing claims, each with a test below:

- ``jax.grad`` through ``propagator(wrt=...)`` matches a central finite
  difference (rtol 1e-4) for a module coefficient, the stepper ``dt``,
  and an initial PROGNOSTIC field — the canonical autodiff invariant,
  now via the public surface (AGENTS.md, "Differentiability policy").
- The forward output equals ``model.advance`` over the same steps
  (fresh stepper state both sides), and a supplied ``state=`` matches
  ``set_state`` + ``advance``.
- ``remat=True`` (checkpointed scan body) reproduces the plain gradient.
- Every non-differentiable ``wrt`` is refused host-side with a taught
  error: an unknown name, an identity-defaulted constant, a
  materialized-owner parameter, and a parameter frozen into ``exp(L
  dt)`` by an exponential stepper — while a non-L parameter and an
  initial field stay differentiable under that same stepper.

Self-contained per the oversized-module shard convention: the small
builders are duplicated rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.model.closures.diffusion import HarmonicDiffusion
from fridom.model.errors import AssemblyError, MissingParameterError
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import Param
from fridom.model.params import TIME_STEP
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 12
L = 1.0
DT = 2e-3
KAPPA = 4e-3
STEPS = 10

# the shallow-water channel (walled y) the eigenbasis engine needs
NC = 16
COMPONENTS = ("u", "v", "p")


# ================================================================
#  Toy model (a differentiable AB3 run)
# ================================================================
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
        """Return a zero tendency (physics rides the mixing closure)."""
        return {name: 0.0 * state[name] for name in ("u", "b", "c")}


class CoreConst(Core):

    """Toy core that references an unprovided defaulted constant."""

    def __init__(self) -> None:
        """Declare a defaulted reference nothing provides (slot None)."""
        self.coeff = Param("toy.coeff", default=1.0)


class Empty(Module):

    """A module that declares no fields (a PROGNOSTIC-free model)."""


def make_grid():
    """Return a tiny periodic (x, z) grid."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=True, name=name)
        for name in ("x", "z")))


def make_model(core=Core, kappa=KAPPA, dt=DT):
    """Build the toy model and seed smooth tracers."""
    model = Model(
        grid=make_grid(),
        modules=(core(), HarmonicDiffusion(kappa)),
        time_stepper=AdamBashforth(dt, order=3))
    ax = (np.arange(N) + 0.5) * (L / N)
    x, z = np.meshgrid(ax, ax, indexing="ij")
    model.set_fields(b=np.sin(2 * np.pi * x), c=np.sin(4 * np.pi * z))
    return model


def state_sq(final):
    """Sum of squares of every final-state field (a smooth loss)."""
    return sum(jnp.sum(f.data ** 2) for f in final.state)


def central_fd(loss, x0, eps=1e-4):
    """Central finite difference of ``loss`` at ``x0`` (relative step)."""
    h = eps * (abs(float(x0)) if float(x0) != 0.0 else 1.0)
    return (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)


# ================================================================
#  Shallow-water channel builders (materialized + frozen-L cases)
# ================================================================
def sw_channel():
    """Return a periodic-x / walled-y shallow-water channel."""
    mx = IntervalMesh(NC, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(NC, (0.0, 1.0), periodic=False, name="y")
    return fr.spatial.Grid((mx, my))


def sw_model(grid, stepper, *, filtered):
    """Assemble a rotating shallow-water channel; filtered drops L."""
    extra = {"term_filter": ~terms.linear} if filtered else {}
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=stepper, **extra)


@pytest.fixture(scope="module")
def sw_grid():
    """Return the shared shallow-water channel grid."""
    return sw_channel()


@pytest.fixture(scope="module")
def sw_basis(sw_grid):
    """Return the UNFILTERED model's eigenbasis: the operator L."""
    return sw.eigenbasis(sw_model(
        sw_grid, AdamBashforth(1e-3, order=3), filtered=False))


# ================================================================
#  (i) grad vs central FD — module coefficient, dt, initial field
# ================================================================
def test_grad_wrt_module_parameter_matches_fd():
    """Match d loss / d(mixing.kappa) against a central FD (finite)."""
    run = make_model().propagator(wrt=("mixing.kappa",), steps=STEPS)
    k0 = jnp.asarray(KAPPA)

    def loss(kappa):
        return state_sq(run((kappa,)))

    grad = float(jax.grad(loss)(k0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(central_fd(loss, k0), rel=1e-4)


def test_grad_wrt_time_step_matches_fd():
    """Match d loss / d(TIME_STEP) (the stepper leaf) against an FD."""
    run = make_model().propagator(wrt=(TIME_STEP,), steps=STEPS)
    dt0 = jnp.asarray(DT)

    def loss(dt):
        return state_sq(run((dt,)))

    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert grad == pytest.approx(central_fd(loss, dt0), rel=1e-4)


def test_grad_wrt_initial_field_matches_fd_directionally():
    """The IC gradient's projection onto a random direction matches FD."""
    model = make_model()
    run = model.propagator(wrt=("b",), steps=STEPS)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return state_sq(run((field,)))

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b0.shape), dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  (ii) forward parity with advance (state=None and state=)
# ================================================================
def test_forward_output_matches_advance():
    """A wrt=() propagator run reproduces model.advance to machine eps."""
    advanced = make_model()
    advanced.advance(STEPS)
    want = {c: np.asarray(advanced.state[c].data)
            for c in ("u", "b", "c")}

    run = make_model().propagator(wrt=(), steps=STEPS)
    out = run(())
    got = {c: np.asarray(out.state[c].data) for c in ("u", "b", "c")}

    for c in ("u", "b", "c"):
        np.testing.assert_allclose(got[c], want[c], rtol=1e-11, atol=1e-12)


def test_state_argument_matches_set_state_then_advance():
    """A supplied (partial) state= splices in like set_state + advance."""
    model = make_model()
    run = model.propagator(wrt=(), steps=STEPS)
    # a PARTIAL PROGNOSTIC state (only b, c) on the model's OWN grid:
    # the absent u must fall through to the base carry both sides.
    # blank_state() is the State factory recipe: a fresh PROGNOSTIC
    # scaffold at declared defaults (04 section 6.1), filled per name.
    rng = np.random.default_rng(3)
    base = model.blank_state()
    external = fr.spatial.VectorField({
        c: base[c].with_data(jnp.asarray(
            rng.standard_normal(np.asarray(base[c].data).shape)))
        for c in ("b", "c")})

    # the propagator snapshotted its carry at build; state= overrides
    got = run((), state=external)
    got = {c: np.asarray(got.state[c].data) for c in ("u", "b", "c")}

    model.set_state(external)
    model.advance(STEPS)
    want = {c: np.asarray(model.state[c].data) for c in ("u", "b", "c")}

    for c in ("u", "b", "c"):
        np.testing.assert_allclose(got[c], want[c], rtol=1e-11, atol=1e-12)


# ================================================================
#  (iii) remat: gradient equality and finiteness
# ================================================================
def test_remat_gradient_matches_plain_and_is_finite():
    """Checkpointing the scan body leaves the gradient unchanged."""
    k0 = jnp.asarray(KAPPA)
    plain = make_model().propagator(wrt=("mixing.kappa",), steps=STEPS)
    checkpointed = make_model().propagator(
        wrt=("mixing.kappa",), steps=STEPS, remat=True)

    g_plain = float(jax.grad(lambda k: state_sq(plain((k,))))(k0))
    g_remat = float(jax.grad(lambda k: state_sq(checkpointed((k,))))(k0))
    assert np.isfinite(g_remat)
    assert g_remat == pytest.approx(g_plain, rel=1e-10)


# ================================================================
#  (iv) argument validation
# ================================================================
@pytest.mark.parametrize(
    "steps",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative"),
     pytest.param(True, id="bool"), pytest.param(2.5, id="float")])
def test_rejects_non_positive_steps(steps):
    """Reject a steps that is not a positive int (bool/float/0/-1)."""
    with pytest.raises(ValueError, match="positive int"):
        make_model().propagator(wrt=(), steps=steps)


def test_rejects_theta_length_mismatch():
    """The run callable checks theta's length against wrt."""
    run = make_model().propagator(wrt=("mixing.kappa",), steps=3)
    with pytest.raises(ValueError, match="expected 1 theta"):
        run((1.0, 2.0))


def test_rejects_prognostic_free_model():
    """A stage/field-free composition has nothing to propagate."""
    model = Model(
        grid=Grid((IntervalMesh(8, (0.0, 1.0), periodic=True, name="x"),)),
        modules=(Empty(),),
        time_stepper=AdamBashforth(1e-3, order=3))
    with pytest.raises(NotImplementedError, match="PROGNOSTIC state"):
        model.propagator(wrt=(), steps=3)


# ================================================================
#  (v) taught refusals
# ================================================================
def test_rejects_unknown_wrt_name():
    """An unknown wrt name lists the valid parameters and fields."""
    with pytest.raises(MissingParameterError,
                       match="neither a bound parameter"):
        make_model().propagator(wrt=("nope.nope",), steps=STEPS)


def test_rejects_identity_defaulted_constant():
    """A slot-None constant has no leaf to differentiate."""
    with pytest.raises(AssemblyError,
                       match="identity-defaulted constant"):
        make_model(CoreConst).propagator(wrt=("toy.coeff",), steps=STEPS)


def test_rejects_materialized_owner_parameter(sw_grid):
    """coriolis.f0 materializes f_coriolis at assembly — refused."""
    model = sw_model(sw_grid, AdamBashforth(1e-3, order=3), filtered=False)
    with pytest.raises(AssemblyError, match="materializes"):
        model.propagator(wrt=("coriolis.f0",), steps=STEPS)


def test_rejects_frozen_linear_operator_parameter(sw_grid, sw_basis):
    """Under ETDRK4, coriolis.f0 feeds the frozen L — refused."""
    model = sw_model(
        sw_grid, fr.model.time_steppers.ETDRK4(0.2 / NC, sw_basis),
        filtered=True)
    with pytest.raises(AssemblyError, match="freezes into an eigenbasis"):
        model.propagator(wrt=("coriolis.f0",), steps=5)


def test_allows_non_l_parameter_and_ic_under_etdrk4(sw_grid, sw_basis):
    """Keep dt and an initial field differentiable under a frozen L."""
    model = sw_model(
        sw_grid, fr.model.time_steppers.ETDRK4(0.2 / NC, sw_basis),
        filtered=True)
    model.set_state(sw.random_state(sw_basis, "vortical", seed=7))

    run_dt = model.propagator(wrt=(TIME_STEP,), steps=5)
    grad_dt = float(jax.grad(lambda d: sum(
        jnp.sum(f.data ** 2) for f in run_dt((d,)).state))(
            jnp.asarray(0.2 / NC)))
    assert np.isfinite(grad_dt)

    run_u = model.propagator(wrt=("u",), steps=5)
    u0 = model._carry.state["u"].storage
    grad_u = np.asarray(jax.grad(lambda field: sum(
        jnp.sum(f.data ** 2) for f in run_u((field,)).state))(u0))
    assert bool(np.all(np.isfinite(grad_u)))
