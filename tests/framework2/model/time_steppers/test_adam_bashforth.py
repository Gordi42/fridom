"""Tests for AdamBashforth (time_steppers/adam_bashforth.py).

Covers the coefficient rows (all orders, the eps'd order-2 row), the
order-2-only eps rule, the dense zero-padded warm-up table and its
saturating counter, eager integration bitwise against a hand-rolled
numpy reference implementing the same premultiplied-row arithmetic
in the same accumulation order (forward and backward dt), the
structural newest-first ring shift, pre-tick tendency evaluation,
CONSTRAINT placement, jit + lax.scan warm-up crossing with a single
compile, the host-side dispersion analysis, and the ABState pytree.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.model.clock import Clock
from fridom.framework2.model.composer import TendencyComposer
from fridom.framework2.model.declarations import Lifecycle
from fridom.framework2.model.stages import Stage, StageKind
from fridom.framework2.model.terms import (
    TERM_ATTRIBUTE,
    Treatment,
    term,
)
from fridom.framework2.model.time_steppers.adam_bashforth import (
    ABState,
    AdamBashforth,
)

LAM = 0.7  # the linear-decay rate of the reference problem


# ================================================================
#  Fakes (duck-typed composer inputs; the test_schedule pattern)
# ================================================================
class Record(NamedTuple):
    name: str
    space: object
    lifecycle: Lifecycle
    owner: int


class FakeFieldTable:
    def __init__(self, grid, records):
        self.grid = grid
        self._records = tuple(records)

    def __iter__(self):
        return iter(self._records)


class Decay:

    """One @fr.term linear-decay tendency: du/dt = -lam * u."""

    lam = LAM

    @term(name="decay")
    def du(self, state, _ctx):
        return {"u": state["u"] * (-self.lam)}


class TimeForcing:

    """A forcing reading the stage time: du/dt = t."""

    @term(name="force")
    def df(self, state, ctx):
        u = state["u"]
        return {"u": u.with_data(
            jnp.broadcast_to(ctx.clock.time, u.data.shape))}


class Projector:
    def project(self, state, _ctx):
        return {"u": state["u"] * 0.5}


class Subcycler:
    def sub(self, state, _ctx):
        return {"u": state["u"]}


# ================================================================
#  Fixtures and helpers
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


@pytest.fixture
def field_table(grid, mx):
    return FakeFieldTable(grid, (
        Record("u", mx.center, Lifecycle.PROGNOSTIC, 0),
    ))


def term_of(method):
    """Return the TendencyTerm record stamped by @fr.term."""
    return getattr(method, TERM_ATTRIBUTE)


def build_schedule(field_table, modules, terms, stepper, stages=()):
    """Build a real wave-3 Schedule over the fakes."""
    return TendencyComposer(
        field_table=field_table,
        modules=modules,
        terms=terms,
        stages=stages,
        time_stepper=stepper,
        binding_table=None,
    ).schedule


def decay_schedule(field_table, stepper):
    modules = (Decay(),)
    schedule = build_schedule(
        field_table, modules, ((0, term_of(Decay.du)),), stepper)
    return schedule, modules


def make_state(field_table, data):
    (record,) = tuple(field_table)
    return VectorField({
        "u": field_table.grid.create_field(
            record.space, data=jnp.asarray(data), name="u")})


@pytest.fixture
def u0():
    return np.linspace(0.5, 1.5, 8)


def run_eager(stepper, schedule, modules, state, steps):
    """Step eagerly from a fresh init; returns (sst, state, clock)."""
    stepper_state = stepper.init(state)
    clock = Clock()
    for _ in range(steps):
        bound = schedule.bind(modules)
        stepper_state, state, clock = stepper.step(
            stepper_state, state, bound, clock)
    return stepper_state, state, clock


def reference_ab(u0, lam, dt, table, steps):
    """Integrate du/dt = -lam*u with a hand-rolled numpy AB.

    The same premultiplied-row arithmetic in the same
    (ascending-j, newest-first) accumulation order as the stepper.
    """
    u = np.asarray(u0, dtype=np.float64).copy()
    order = len(table)
    ring = [np.zeros_like(u) for _ in range(order)]
    rows = np.asarray(table, dtype=np.float64)
    level = 0
    for _ in range(steps):
        tendency = np.zeros_like(u) + u * (-lam)
        ring = [tendency, *ring[:-1]]
        weights = rows[level] * np.float64(dt)  # premultiply first
        increment = weights[0] * ring[0]
        for j in range(1, order):               # ascending j
            increment = increment + weights[j] * ring[j]
        u = u + increment
        level = min(level + 1, order - 1)
    return u


# ================================================================
#  Coefficient rows and the warm-up table
# ================================================================
def test_order1_table():
    assert AdamBashforth(1.0, order=1).table == ((1.0,),)


def test_order2_table_carries_the_eps_damper():
    stepper = AdamBashforth(1.0, order=2)
    assert stepper.eps == 0.01  # the parity default
    assert stepper.table == (
        (1.0, 0.0),
        (3 / 2 + 0.01, -1 / 2 - 0.01),
    )


def test_order2_explicit_eps():
    stepper = AdamBashforth(1.0, order=2, eps=0.25)
    assert stepper.eps == 0.25
    assert stepper.table[1] == (3 / 2 + 0.25, -1 / 2 - 0.25)


def test_order3_table_warms_up_through_textbook_ab2():
    # order >= 3 warm-up uses textbook AB2 [3/2, -1/2] — the
    # deliberate startup-only delta vs the old code (02_rules)
    assert AdamBashforth(1.0, order=3).table == (
        (1.0, 0.0, 0.0),
        (3 / 2, -1 / 2, 0.0),
        (23 / 12, -4 / 3, 5 / 12),
    )


def test_order4_table():
    assert AdamBashforth(1.0, order=4).table == (
        (1.0, 0.0, 0.0, 0.0),
        (3 / 2, -1 / 2, 0.0, 0.0),
        (23 / 12, -4 / 3, 5 / 12, 0.0),
        (55 / 24, -59 / 24, 37 / 24, -3 / 8),
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_table_is_dense_order_by_order(order):
    table = AdamBashforth(1.0, order=order).table
    assert len(table) == order
    assert all(len(row) == order for row in table)


# ================================================================
#  The eps rule (order-2-only) and order validation
# ================================================================
@pytest.mark.parametrize("order", [1, 3, 4])
def test_eps_rejected_off_order_2(order):
    with pytest.raises(ValueError, match="order-2-only"):
        AdamBashforth(1.0, order=order, eps=0.01)


@pytest.mark.parametrize("order", [1, 3, 4])
def test_eps_is_none_off_order_2(order):
    assert AdamBashforth(1.0, order=order).eps is None


@pytest.mark.parametrize("order", [0, 5, -1])
def test_order_out_of_range(order):
    with pytest.raises(ValueError, match="orders 1 to 4"):
        AdamBashforth(1.0, order=order)


def test_supported_treatments_is_explicit_only():
    assert (AdamBashforth(1.0).supported_treatments
            == frozenset({Treatment.EXPLICIT}))


# ================================================================
#  Statics, fingerprint, pytree
# ================================================================
def test_fingerprint_token_carries_order_and_eps():
    assert AdamBashforth(1.0, order=3).fingerprint_token() == (
        "AdamBashforth", ("order", 3), ("eps", None))
    assert AdamBashforth(1.0, order=2).fingerprint_token() == (
        "AdamBashforth", ("order", 2), ("eps", 0.01))


def test_fingerprint_token_excludes_dt():
    assert (AdamBashforth(1.0, order=3).fingerprint_token()
            == AdamBashforth(-60.0, order=3).fingerprint_token())


def test_stepper_pytree_round_trip():
    stepper = AdamBashforth(60.0, order=2, eps=0.05)
    leaves, treedef = jax.tree_util.tree_flatten(stepper)
    assert len(leaves) == 1  # dt is the only dynamic leaf
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.order == 2
    assert rebuilt.eps == 0.05
    assert rebuilt.table == stepper.table
    assert float(rebuilt.dt) == 60.0


def test_repr_smoke():
    assert "order=3" in repr(AdamBashforth(60.0, order=3))
    assert "eps" in repr(AdamBashforth(60.0, order=2))


# ================================================================
#  init — the fresh (re-warmed) carry
# ================================================================
def test_init_builds_a_zeroed_ring(field_table, u0):
    stepper = AdamBashforth(0.5, order=3)
    template = make_state(field_table, u0)
    stepper_state = stepper.init(template)
    assert isinstance(stepper_state, ABState)
    assert len(stepper_state.history) == 3
    for entry in stepper_state.history:
        assert entry.component_names == ("u",)
        assert np.all(np.asarray(entry["u"].data) == 0.0)
    assert stepper_state.warmup.dtype == jnp.int32
    assert int(stepper_state.warmup) == 0
    # the template is untouched (functional init)
    assert np.array_equal(np.asarray(template["u"].data), u0)


def test_ab_state_is_frozen(field_table, u0):
    stepper_state = AdamBashforth(0.5, order=2).init(
        make_state(field_table, u0))
    with pytest.raises(AttributeError, match="frozen"):
        stepper_state.warmup = jnp.asarray(1, dtype=jnp.int32)
    with pytest.raises(AttributeError, match="frozen"):
        del stepper_state.history


def test_ab_state_jaxify_round_trip(field_table, u0):
    stepper_state = AdamBashforth(0.5, order=2).init(
        make_state(field_table, u0))
    leaves, treedef = jax.tree_util.tree_flatten(stepper_state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is ABState
    assert len(rebuilt.history) == 2
    assert int(rebuilt.warmup) == 0


# ================================================================
#  Eager integration — bitwise against the numpy reference
# ================================================================
@pytest.mark.parametrize("direction", [1.0, -1.0],
                         ids=["forward", "backward"])
@pytest.mark.parametrize(("order", "steps"),
                         [(1, 2), (2, 3), (3, 5), (4, 6)])
def test_eager_integration_bitwise(field_table, u0, order, steps,
                                   direction):
    dt = 0.3 * direction  # signed dt flows through everything
    stepper = AdamBashforth(dt, order=order)
    schedule, modules = decay_schedule(field_table, stepper)
    state = make_state(field_table, u0)
    _, state, clock = run_eager(stepper, schedule, modules, state,
                                steps)
    expected = reference_ab(u0, LAM, dt, stepper.table, steps)
    assert np.array_equal(np.asarray(state["u"].data), expected)
    assert int(clock.it) == steps  # it increments both directions
    assert float(clock.elapsed) == pytest.approx(steps * dt)


def test_warmup_counter_saturates(field_table, u0):
    stepper = AdamBashforth(0.1, order=3)
    schedule, modules = decay_schedule(field_table, stepper)
    state = make_state(field_table, u0)
    stepper_state = stepper.init(state)
    clock = Clock()
    seen = []
    for _ in range(5):
        stepper_state, state, clock = stepper.step(
            stepper_state, state, schedule.bind(modules), clock)
        seen.append(int(stepper_state.warmup))
    assert seen == [1, 2, 2, 2, 2]


def test_ring_shift_is_newest_first(field_table, u0):
    stepper = AdamBashforth(0.25, order=2)
    schedule, modules = decay_schedule(field_table, stepper)
    state = make_state(field_table, u0)
    stepper_state = stepper.init(state)
    clock = Clock()
    stepper_state, state1, clock = stepper.step(
        stepper_state, state, schedule.bind(modules), clock)
    # history[0] is the newest tendency (of the pre-step state);
    # history[1] is still the zero-initialized entry
    first = np.zeros_like(u0) + u0 * (-LAM)
    assert np.array_equal(
        np.asarray(stepper_state.history[0]["u"].data), first)
    assert np.all(
        np.asarray(stepper_state.history[1]["u"].data) == 0.0)
    stepper_state, _, clock = stepper.step(
        stepper_state, state1, schedule.bind(modules), clock)
    second = (np.zeros_like(u0)
              + np.asarray(state1["u"].data) * (-LAM))
    assert np.array_equal(
        np.asarray(stepper_state.history[0]["u"].data), second)
    assert np.array_equal(
        np.asarray(stepper_state.history[1]["u"].data), first)


def test_tendency_is_evaluated_at_the_pre_tick_time(field_table):
    dt = 0.5
    stepper = AdamBashforth(dt, order=1)
    modules = (TimeForcing(),)
    schedule = build_schedule(
        field_table, modules, ((0, term_of(TimeForcing.df)),),
        stepper)
    state = make_state(field_table, np.zeros(8))
    stepper_state = stepper.init(state)
    clock = Clock()
    stepper_state, state, clock = stepper.step(
        stepper_state, state, schedule.bind(modules), clock)
    # the forcing read t = 0 (pre-tick), not t = dt
    assert np.all(np.asarray(state["u"].data) == 0.0)
    assert float(clock.time) == pytest.approx(dt)
    stepper_state, state, clock = stepper.step(
        stepper_state, state, schedule.bind(modules), clock)
    assert np.allclose(np.asarray(state["u"].data), dt * dt)


def test_constraint_stages_run_after_the_advance(field_table, u0):
    dt = 0.25
    stepper = AdamBashforth(dt, order=1)
    modules = (Decay(), Projector())
    schedule = build_schedule(
        field_table, modules, ((0, term_of(Decay.du)),), stepper,
        stages=((1, Stage(kind=StageKind.CONSTRAINT,
                          fn=Projector.project, name="project")),))
    state = make_state(field_table, u0)
    _, state, _ = run_eager(stepper, schedule, modules, state, 1)
    expected = (u0 + dt * (np.zeros_like(u0) + u0 * (-LAM))) * 0.5
    assert np.allclose(np.asarray(state["u"].data), expected)


def test_advance_stages_still_gate_on_wave5(field_table, u0):
    # compositions WITH module-owned ADVANCE stages hit the wave-3
    # BoundSchedule stub; compositions without them skip the group
    stepper = AdamBashforth(0.25, order=1)
    modules = (Decay(), Subcycler())
    schedule = build_schedule(
        field_table, modules, ((0, term_of(Decay.du)),), stepper,
        stages=((1, Stage(kind=StageKind.ADVANCE, fn=Subcycler.sub,
                          name="sub", advances=("u",))),))
    state = make_state(field_table, u0)
    stepper_state = stepper.init(state)
    with pytest.raises(NotImplementedError, match=r"2\.5"):
        stepper.step(stepper_state, state, schedule.bind(modules),
                     Clock())


# ================================================================
#  jit + lax.scan — one compile, warm-up crossing inside the scan
# ================================================================
def test_jit_scan_compiles_once_and_crosses_warmup(
        field_table, u0, compile_counter):
    steps = 5
    stepper = AdamBashforth(0.2, order=3)
    schedule, modules = decay_schedule(field_table, stepper)

    def run(stepper, carry):
        def body(carry, _):
            state, stepper_state, clock = carry
            bound = schedule.bind(modules)
            stepper_state, state, clock = stepper.step(
                stepper_state, state, bound, clock)
            return (state, stepper_state, clock), None

        return jax.lax.scan(body, carry, None, length=steps)[0]

    run_jit = jax.jit(run)
    # normalize the input sharding SPECS to the jit-output form
    # (host-born fields carry P(None,) vs output P() — equivalent
    # on one device but a distinct jit cache key; the chunk
    # runner's carry discipline is wave 4.2's)
    canonical = jax.jit(lambda tree: tree)

    state = make_state(field_table, u0)
    carry = canonical((state, stepper.init(state), Clock()))
    stepper_in = canonical(stepper)
    compile_counter.reset()

    final_state, final_sst, final_clock = run_jit(stepper_in, carry)
    compiles = compile_counter.count
    assert compiles >= 1

    # dt is a dynamic leaf: a dt sweep shares the compile
    other = AdamBashforth(0.1, order=3)
    state2 = make_state(field_table, u0)
    run_jit(canonical(other),
            canonical((state2, other.init(state2), Clock())))
    assert compile_counter.count == compiles

    # a mid/post-warm-up carry re-enters the SAME trace (the
    # bitwise mid-warm-up restart obligation)
    run_jit(stepper_in, (final_state, final_sst, final_clock))
    assert compile_counter.count == compiles

    # the scan crossed the warm-up inside one trace
    assert int(final_sst.warmup) == 2
    assert int(final_clock.it) == steps
    _, eager_state, _ = run_eager(stepper, schedule, modules,
                                  make_state(field_table, u0), steps)
    assert np.allclose(np.asarray(final_state["u"].data),
                       np.asarray(eager_state["u"].data),
                       rtol=1e-12, atol=0.0)


# ================================================================
#  time_discretization_effect — host-side dispersion analysis
# ================================================================
def test_dispersion_satisfies_the_ab2_stability_polynomial():
    dt = 0.1
    stepper = AdamBashforth(dt, order=2)  # eps'd full row
    c0, c1 = stepper.table[-1]
    omega = np.asarray([0.5, 1.0, 2.0])
    discrete = stepper.time_discretization_effect(omega)
    x = np.exp(-1j * discrete * dt)
    residual = x**2 - x + 1j * omega * dt * (c0 * x + c1)
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_dispersion_small_omega_limit():
    stepper = AdamBashforth(0.01, order=3)
    omega = np.asarray([0.05, 0.1])
    discrete = stepper.time_discretization_effect(omega)
    assert np.allclose(discrete.real, omega, rtol=1e-6)
    assert np.allclose(discrete.imag, 0.0, atol=1e-8)


def test_dispersion_dt_override_matches_a_rebuilt_stepper():
    omega = np.linspace(0.1, 1.0, 4)
    base = AdamBashforth(1.0, order=2)
    other = AdamBashforth(0.25, order=2)
    assert np.allclose(
        base.time_discretization_effect(omega, dt=0.25),
        other.time_discretization_effect(omega))


def test_dispersion_preserves_shape_and_is_complex():
    stepper = AdamBashforth(0.1, order=4)
    result = stepper.time_discretization_effect(np.ones((2, 3)))
    assert result.shape == (2, 3)
    assert result.dtype == np.complex128
