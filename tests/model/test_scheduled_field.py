"""Scheduled fields: helper, ProfileFunction, and the SELF_UPDATE recompute.

The general half of the time-dependent-fields plan (TDF-D1/D2/D3/D8):
a static spatial law ``fn(*coords, t, *params)`` sampled at stage time
into an AUXILIARY field its owner rewrites every substage through a
SELF_UPDATE stage. These tests exercise the machinery on a TOY module
independent of the Coriolis / csqr consumers: the sampled value at
stage time equals the analytic law, the carry treedef stays scan-stable,
sweeping the law parameters never recompiles (they are dynamic leaves),
the recompute is device-count invariant, a marked field with no
SELF_UPDATE stage fails the lint, a frozen-``L`` stepper refuses the
marked field, and ``jax.grad`` flows through a param leaf (TDF-D8).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import jaxify
from fridom.model.assembly import assemble
from fridom.model.declarations import FieldDeclaration, Lifecycle
from fridom.model.errors import (
    AssemblyError,
    TimeDependentLinearOperatorError,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.scheduled_field import (
    ProfileFunction,
    profile_coords,
    sample_law,
)
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import Treatment, term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated, Profile

N = 8
DT = 5e-3


def _x_coords():
    """Return the toy grid's x cell-centre coordinates (periodic [0, 1])."""
    return (np.arange(N) + 0.5) * (1.0 / N)


# ================================================================
#  ProfileFunction: the user surface (static law, dynamic leaves)
# ================================================================
def _law(x, t, a):
    """Return ``a * (x + t)`` (a pure spatial-and-temporal law)."""
    return a * (x + t)


def test_profile_function_rejects_a_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        ProfileFunction(3.0)


def test_profile_function_params_are_dynamic_leaves():
    pf = ProfileFunction(_law, params=(2.0,))
    assert isinstance(pf.params, tuple)
    assert jnp.asarray(pf.params[0]).shape == ()
    # the law is static aux; the params are dynamic pytree leaves
    leaves = jax.tree_util.tree_leaves(pf)
    assert len(leaves) == 1
    np.testing.assert_allclose(float(leaves[0]), 2.0)


def test_profile_function_repr_round_trips_the_leaves():
    pf = ProfileFunction(_law, params=(2.0,))
    assert repr(pf) == "ProfileFunction(_law, params=(2.0))"


def test_profile_function_has_no_field_or_array_arithmetic():
    # a consumer forgetting to sample must fail loudly (no operators)
    pf = ProfileFunction(_law, params=(1.0,))

    class FieldLike:
        pass

    with pytest.raises(TypeError):
        pf * FieldLike()
    with pytest.raises(TypeError):
        pf + FieldLike()
    with pytest.raises(TypeError):
        pf * 2.0


def test_sample_evaluates_the_law_at_coords_and_time():
    pf = ProfileFunction(_law, params=(2.0,))
    x = jnp.linspace(0.0, 1.0, N)
    got = pf.sample((x,), 0.3, (N,))
    np.testing.assert_allclose(np.asarray(got), 2.0 * (np.asarray(x) + 0.3),
                               atol=1e-14)


def test_sample_broadcasts_a_space_constant_law():
    # a law constant in the coordinate broadcasts to the field shape
    pf = ProfileFunction(lambda x, t: t)  # noqa: ARG005
    got = pf.sample((jnp.zeros(N),), 0.7, (N,))
    np.testing.assert_allclose(np.asarray(got), np.full(N, 0.7), atol=1e-14)


# ================================================================
#  sample_law: the shared core (value-only + optional jvp _dot)
# ================================================================
def test_sample_law_value_only_broadcasts():
    got = sample_law(lambda c, t: c[0] + t, (jnp.arange(N, dtype=float),),
                     0.5, (N,))
    np.testing.assert_allclose(np.asarray(got),
                               np.arange(N) + 0.5, atol=1e-13)


def test_sample_law_derivative_returns_value_and_dot():
    # d/dt [c*t^2] = 2*c*t: the jvp core returns both, broadcast to shape
    val, dot = sample_law(lambda c, t: c * t ** 2, 3.0, 2.0, (4,),
                          derivative=True)
    np.testing.assert_allclose(np.asarray(val), np.full(4, 12.0), atol=1e-13)
    np.testing.assert_allclose(np.asarray(dot), np.full(4, 12.0), atol=1e-13)


# ================================================================
#  A toy module: g(x,t) = law(x,t) rewritten each substage; du/dt = g
# ================================================================
@partial(jaxify, dynamic=("law",))
class ScheduledForcing(Module):

    """One PROGNOSTIC ``u`` forced by a scheduled AUXILIARY ``g(x,t)``.

    ``g`` is a ``time_dependent`` AUXILIARY field rewritten every
    substage by a SELF_UPDATE stage from the ``ProfileFunction`` law
    (the general recompute path); ``du/dt = g`` reads the fresh
    stage-time value. The SELF_UPDATE writes ``g`` from raw sampled data
    (``with_data``, halo-trace exempt, V-N2); ``g.to(u)`` is a same-node
    identity (reach 0), so the zero substitute is exact.
    """

    #: the SELF_UPDATE reads the clock and rewrites g from raw data
    #: (halo-trace exempt, V-N2); the g.to(u) lift is reach 0
    extra_halo = HaloSpec({})

    def __init__(self, law):
        """Store the ProfileFunction law (a dynamic pytree child)."""
        self.law = law

    @property
    def field_declarations(self):
        """PROGNOSTIC ``u`` plus the scheduled AUXILIARY ``g``."""
        return (
            FieldDeclaration("u", space=Collocated(), long_name="Forced"),
            FieldDeclaration(
                "g", space=Profile("x"), lifecycle=Lifecycle.AUXILIARY,
                default=self._g_default, long_name="Scheduled forcing",
                time_dependent=True),
        )

    def _g_default(self, grid, space):
        """Owner-method default: sample the law at t = 0."""
        coords = profile_coords(grid, space, ("x",))
        data = self.law.sample(coords, 0.0, space.shape)
        return grid.create_field(space, data=data, name="g")

    @property
    def stages(self):
        """The per-substage recompute (SELF_UPDATE, S1)."""
        return (Stage(kind=StageKind.SELF_UPDATE, fn="_update_g",
                      name="scheduled_g", reads=("g",), writes=("g",)),)

    def _update_g(self, state, ctx):
        """Re-evaluate the law at the substage clock and rewrite ``g``."""
        time = getattr(ctx.clock, "time", ctx.clock)
        field = state["g"]
        coords = profile_coords(field.grid, field.function_space, ("x",))
        value = self.law.sample(coords, time, field.function_space.shape)
        return {"g": field.with_data(value)}

    @term(advances=("u",), linear=True, linear_fields=("g",))
    def force(self, state, _ctx):
        """``du/dt = g(x,t)`` from the fresh stage-time field."""
        return {"u": state["g"].to(state["u"])}


def _grid(device_ids=None):
    """Return a small periodic-x grid (one blocked factor to shard)."""
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),),
                device_ids=device_ids)


def _model(law, *, dt=DT, grid=None):
    """Return a one-module model forced by the scheduled field."""
    return Model(
        grid=_grid() if grid is None else grid,
        modules=(ScheduledForcing(law),),
        time_stepper=AdamBashforth(dt, order=3))


def _g_values(model):
    """Return the ``g`` field values (raveled)."""
    return np.asarray(model.state["g"].data).ravel()


# ================================================================
#  Stage-time correctness against a hand-stepped AB3 oracle
# ================================================================
def test_g_materializes_the_law_at_t0():
    model = _model(ProfileFunction(_law, params=(1.7,)))
    np.testing.assert_allclose(_g_values(model), 1.7 * _x_coords(),
                               atol=1e-14)


def test_update_tracks_the_stage_time():
    model = _model(ProfileFunction(_law, params=(1.0,)))
    model.advance(3)
    # the SELF_UPDATE of step k runs at start-of-step time (k-1)*dt
    np.testing.assert_allclose(_g_values(model), _x_coords() + 2 * DT,
                               atol=1e-13)


def test_matches_a_hand_stepped_ab3_oracle():
    """The scheduled forcing advances under AB3 like a hand oracle.

    ``du/dt = g(x, t_n)`` is a state-independent forcing read at the
    pre-tick clock ``t_n = n*dt`` (adam_bashforth.py step()); the AB
    weights do not depend on the forcing, so hand-stepping the exact AB3
    combination of ``law(x, t_n)`` reproduces the run.
    """
    a = 1.3
    order, steps = 3, 4
    model = _model(ProfileFunction(_law, params=(a,)))
    x = _x_coords()
    u0 = np.asarray(model.state["u"].data).ravel().copy()
    model.advance(steps)
    got = np.asarray(model.state["u"].data).ravel()

    table = ((1.0, 0.0, 0.0), (1.5, -0.5, 0.0), (23 / 12, -4 / 3, 5 / 12))
    ring = [np.zeros_like(u0) for _ in range(order - 1)]
    u = u0.copy()
    for n in range(steps):
        f_n = a * (x + n * DT)
        levels = (f_n, *ring)
        weights = table[min(n, order - 1)]
        u = u + DT * sum(weights[j] * levels[j] for j in range(order))
        ring = (f_n, *ring)[:order - 1]
    np.testing.assert_allclose(got, u, rtol=1e-11, atol=1e-13)


def test_scheduled_run_differs_from_the_frozen_law():
    # sanity: the time dependence actually drives the solution
    model = _model(ProfileFunction(_law, params=(2.0,)))
    model.advance(6)
    scheduled = np.asarray(model.state["u"].data)
    frozen = _model(ProfileFunction(
        lambda x, t, a: a * x, params=(2.0,)))  # noqa: ARG005
    frozen.advance(6)
    assert not np.allclose(scheduled, np.asarray(frozen.state["u"].data))


# ================================================================
#  Treedef stability: the law rides the carry, params keep their shape
# ================================================================
def test_carry_treedef_is_stable_across_steps():
    model = _model(ProfileFunction(_law, params=(1.0,)))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(4)
    after = jax.tree_util.tree_structure(model._carry)
    assert after == before


# ================================================================
#  Zero-recompile param sweeps; a distinct law recompiles once
# ================================================================
def test_param_sweeps_do_not_recompile(compile_counter):
    model = _model(ProfileFunction(_law, params=(1.0,)))
    coords = profile_coords(
        model.grid, model.state["g"].function_space, ("x",))
    shape = model.state["g"].function_space.shape

    @partial(jax.jit, static_argnums=(2,))
    def sample_value(law, t, shape):
        return law.sample(coords, t, shape)

    # build the swept laws BEFORE reset(): eager param coercion traces
    warm = ProfileFunction(_law, params=(0.5,))
    swept = [ProfileFunction(_law, params=(2.0,)),
             ProfileFunction(_law, params=(-1.0,))]
    sample_value(warm, 0.4, shape).block_until_ready()

    compile_counter.reset()
    for law in swept:
        sample_value(law, 0.4, shape).block_until_ready()
    assert compile_counter.count == 0


def test_a_distinct_law_recompiles_once(compile_counter):
    coords = (jnp.linspace(0.0, 1.0, N),)

    @partial(jax.jit, static_argnums=(2,))
    def sample_value(law, t, shape):
        return law.sample(coords, t, shape)

    warm = ProfileFunction(_law, params=(1.0,))
    other = ProfileFunction(lambda x, t, a: a * jnp.sin(x + t), params=(1.0,))
    sample_value(warm, 0.4, (N,)).block_until_ready()

    compile_counter.reset()
    sample_value(other, 0.4, (N,)).block_until_ready()  # distinct law
    assert compile_counter.count == 1


# ================================================================
#  Device-count invariance (the pointwise recompute is halo-neutral)
# ================================================================
@pytest.mark.multi_device
def test_recompute_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    pf = ProfileFunction(lambda x, t, a: a * jnp.sin(x + 4.0 * t),
                         params=(0.7,))

    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = _model(pf, grid=_grid(device_ids=device_ids))
        model.advance(5)
        results[tag] = np.asarray(model.state["u"].data)
        if tag == "many":
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    np.testing.assert_allclose(results["many"], results["one"],
                               rtol=1e-11, atol=1e-13)


# ================================================================
#  The TDF-D3 lint and the ETDRK4 refusal
# ================================================================
class _Stepper:

    """Duck-typed explicit stepper (dt leaf + one static)."""

    supported_treatments = frozenset({Treatment.EXPLICIT})

    def __init__(self, dt=DT, order=2):
        self.dt = dt
        self.order = order


class _FrozenLStepper(_Stepper):

    """A stepper that integrates ``L`` from a frozen eigenbasis snapshot."""

    freezes_linear_operator = True


@partial(jaxify, dynamic=())
class _Unwritten(Module):

    """A time_dependent field with NO SELF_UPDATE stage (lint bait)."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated(), long_name="u"),
        FieldDeclaration(
            "g", space=Profile("x"), lifecycle=Lifecycle.AUXILIARY,
            default=1.0, long_name="g", time_dependent=True),
    )

    @term(advances=("u",), linear=True, linear_fields=("g",))
    def force(self, state, _ctx):
        return {"u": state["g"].to(state["u"])}


def test_marked_field_without_a_self_update_stage_fails_the_lint():
    with pytest.raises(AssemblyError,
                       match=r"g \(_Unwritten\).*SELF_UPDATE"):
        assemble(grid=_grid(), modules=(_Unwritten(),),
                 time_stepper=_Stepper())


def test_a_self_update_stage_satisfies_the_lint():
    # the toy WITH the SELF_UPDATE stage assembles cleanly
    model = _model(ProfileFunction(_law, params=(1.0,)))
    assert "g" in model.state


def test_frozen_l_stepper_refuses_the_marked_field():
    with pytest.raises(TimeDependentLinearOperatorError,
                       match=r"g \(ScheduledForcing\)"):
        assemble(grid=_grid(),
                 modules=(ScheduledForcing(ProfileFunction(_law, (1.0,))),),
                 time_stepper=_FrozenLStepper())


# ================================================================
#  TDF-D8: jax.grad through a run w.r.t. a ProfileFunction param leaf
# ================================================================
def _param_leaf(carry):
    """Return the ProfileFunction param leaf on the carried module."""
    module = next(m for m in carry.modules
                  if isinstance(m, ScheduledForcing))
    return module.law.params[0]


def test_grad_wrt_a_profile_function_param_is_finite_and_matches_fd():
    """Grad w.r.t. a law param leaf is finite and matches FD (TDF-D8)."""
    model = _model(ProfileFunction(_law, params=(1.3,)))
    model.set_fields(u=np.sin(2 * np.pi * _x_coords()))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    a0 = _param_leaf(carry)

    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is a0]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 8, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = float(jax.grad(loss)(a0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-4 * abs(float(a0))
    fd = (float(loss(a0 + h)) - float(loss(a0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
