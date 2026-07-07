"""Tests for the explicit Runge-Kutta family (runge_kutta.py).

Covers the ButcherTableau data (every preset exactly), the
embedded-tableau rejection, eager linear-decay integration bitwise
against a hand-rolled numpy reference doing the same premultiplied,
ascending-j arithmetic per tableau, LowStorageRK3 vs
ExplicitRungeKutta(RK3) at the same order of accuracy with the pinned
coefficients fingerprinted, measured convergence order on a smooth
problem (RK2~2, RK3~3, RK4~4), a single jit/scan compile, and the
project-the-state CONSTRAINT placement (RK3 fires the constraint per
stage plus the final combination).
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
from fridom.framework2.model.time_steppers.runge_kutta import (
    ButcherTableau,
    ExplicitRungeKutta,
    LowStorageRK3,
    tableaus,
)

LAM = 0.7  # the linear-decay rate of the reference problem


# ================================================================
#  Fakes (the test_adam_bashforth harness)
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

    """du/dt = -lam * u."""

    lam = LAM

    @term(name="decay")
    def du(self, state, _ctx):
        return {"u": state["u"] * (-self.lam)}


class TimeForcing:

    """du/dt = t (reads the stage clock — the c_i * dt shift)."""

    @term(name="force")
    def df(self, state, ctx):
        u = state["u"]
        return {"u": u.with_data(
            jnp.broadcast_to(ctx.clock.time, u.data.shape))}


class Counter:

    """A CONSTRAINT stage that counts its invocations (no writes)."""

    def __init__(self):
        self.calls = 0

    def project(self, _state, _ctx):
        self.calls += 1
        return {}


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
    return getattr(method, TERM_ATTRIBUTE)


def build_schedule(field_table, modules, terms, stepper, stages=()):
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


def run_eager(stepper, schedule, modules, state, steps):
    stepper_state = stepper.init(state)
    clock = Clock()
    for _ in range(steps):
        stepper_state, state, clock = stepper.step(
            stepper_state, state, schedule.bind(modules), clock)
    return stepper_state, state, clock


@pytest.fixture
def u0():
    return np.linspace(0.5, 1.5, 8)


# ================================================================
#  ButcherTableau data exactness
# ================================================================
def test_euler_tableau():
    assert tableaus.EULER.a == ((0.0,),)
    assert tableaus.EULER.b == (1.0,)
    assert tableaus.EULER.c == (0.0,)
    assert tableaus.EULER.stages == 1


def test_rk2_tableau():
    assert tableaus.RK2.a == ((0.0, 0.0), (1 / 2, 0.0))
    assert tableaus.RK2.b == (0.0, 1.0)
    assert tableaus.RK2.c == (0.0, 1 / 2)


def test_rk3_tableau():
    assert tableaus.RK3.a == (
        (0.0, 0.0, 0.0), (1 / 2, 0.0, 0.0), (-1.0, 2.0, 0.0))
    assert tableaus.RK3.b == (1 / 6, 2 / 3, 1 / 6)
    assert tableaus.RK3.c == (0.0, 1 / 2, 1.0)


def test_rk4_tableau():
    assert tableaus.RK4.a == (
        (0.0, 0.0, 0.0, 0.0), (1 / 2, 0.0, 0.0, 0.0),
        (0.0, 1 / 2, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    assert tableaus.RK4.b == (1 / 6, 1 / 3, 1 / 3, 1 / 6)
    assert tableaus.RK4.c == (0.0, 1 / 2, 1 / 2, 1.0)


def test_rk4_38_tableau():
    assert tableaus.RK4_38.a == (
        (0.0, 0.0, 0.0, 0.0), (1 / 3, 0.0, 0.0, 0.0),
        (-1 / 3, 1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0))
    assert tableaus.RK4_38.b == (1 / 8, 3 / 8, 3 / 8, 1 / 8)
    assert tableaus.RK4_38.c == (0.0, 1 / 3, 2 / 3, 1.0)


@pytest.mark.parametrize(
    "tableau", [tableaus.HEUN_EULER, tableaus.BOGACKI_SHAMPINE,
                tableaus.RKF45])
def test_embedded_tableaus_carry_b_error(tableau):
    assert tableau.is_embedded
    assert tableau.b_error is not None
    assert len(tableau.b_error) == tableau.stages


def test_tableau_is_frozen_and_hashable():
    assert hash(tableaus.RK4) == hash(tableaus.RK4)
    with pytest.raises(Exception):  # noqa: B017,PT011 — dataclass FrozenInstanceError
        tableaus.RK4.b = (1.0,)


def test_tableau_normalizes_lists_to_tuples():
    tab = ButcherTableau(a=[[0.0]], b=[1.0], c=[0.0])
    assert tab.a == ((0.0,),)
    assert isinstance(tab.b, tuple)


# ================================================================
#  Embedded-tableau rejection
# ================================================================
@pytest.mark.parametrize(
    "tableau", [tableaus.HEUN_EULER, tableaus.BOGACKI_SHAMPINE,
                tableaus.RKF45])
def test_embedded_tableau_is_rejected(tableau):
    with pytest.raises(ValueError, match="fixed-step"):
        ExplicitRungeKutta(0.1, tableau)


def test_rejects_non_tableau():
    with pytest.raises(TypeError, match="ButcherTableau"):
        ExplicitRungeKutta(0.1, "rk4")


def test_default_tableau_is_rk4():
    assert ExplicitRungeKutta(0.1).tableau is tableaus.RK4


def test_supported_treatments_is_explicit_only():
    assert (ExplicitRungeKutta(0.1).supported_treatments
            == frozenset({Treatment.EXPLICIT}))
    assert (LowStorageRK3(0.1).supported_treatments
            == frozenset({Treatment.EXPLICIT}))


# ================================================================
#  Statics, fingerprint, pytree
# ================================================================
def test_rk_carries_the_unit_pytree(field_table, u0):
    stepper = ExplicitRungeKutta(0.1, tableaus.RK3)
    assert stepper.init(make_state(field_table, u0)) == ()
    assert LowStorageRK3(0.1).init(make_state(field_table, u0)) == ()


def test_fingerprint_excludes_dt_includes_tableau():
    a = ExplicitRungeKutta(0.1, tableaus.RK4)
    b = ExplicitRungeKutta(-9.0, tableaus.RK4)
    c = ExplicitRungeKutta(0.1, tableaus.RK3)
    assert a.fingerprint_token() == b.fingerprint_token()
    assert a.fingerprint_token() != c.fingerprint_token()


def test_low_storage_rk3_coefficients_are_pinned():
    # the Oceananigans RK3 reference (deviation D-6): fingerprinted
    assert LowStorageRK3(0.1).coefficients == (
        (8 / 15, 0.0, 0.0),
        (5 / 12, -17 / 60, 8 / 15),
        (3 / 4, -5 / 12, 2 / 3),
    )
    assert LowStorageRK3(0.1).fingerprint_token() == (
        "LowStorageRK3",
        ("coefficients", LowStorageRK3(0.1).coefficients))


def test_rk_pytree_round_trip():
    stepper = ExplicitRungeKutta(60.0, tableaus.RK3)
    leaves, treedef = jax.tree_util.tree_flatten(stepper)
    assert len(leaves) == 1  # dt is the only dynamic leaf
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.tableau == tableaus.RK3
    assert float(rebuilt.dt) == 60.0


# ================================================================
#  Eager integration — bitwise against a hand-rolled reference
# ================================================================
def reference_rk(u0, lam, dt, a, b, steps):
    """Integrate du/dt = -lam*u with the same op sequence as step."""
    u = np.asarray(u0, dtype=np.float64).copy()
    stages = len(b)
    for _ in range(steps):
        k = []
        for i in range(stages):
            if i == 0:
                stage = u.copy()
            else:
                inc = (a[i][0] * dt) * k[0]
                for j in range(1, i):
                    inc = inc + (a[i][j] * dt) * k[j]
                stage = u + inc
            k.append(stage * (-lam))
        inc = (b[0] * dt) * k[0]
        for j in range(1, stages):
            inc = inc + (b[j] * dt) * k[j]
        u = u + inc
    return u


@pytest.mark.parametrize("direction", [1.0, -1.0],
                         ids=["forward", "backward"])
@pytest.mark.parametrize(
    "tableau",
    [tableaus.EULER, tableaus.RK2, tableaus.RK3, tableaus.RK4,
     tableaus.RK4_38])
def test_eager_integration_bitwise(field_table, u0, tableau,
                                   direction):
    dt = 0.3 * direction
    stepper = ExplicitRungeKutta(dt, tableau)
    schedule, modules = decay_schedule(field_table, stepper)
    state = make_state(field_table, u0)
    _, state, clock = run_eager(stepper, schedule, modules, state, 4)
    expected = reference_rk(u0, LAM, dt, tableau.a, tableau.b, 4)
    assert np.allclose(np.asarray(state["u"].data), expected,
                       rtol=1e-13, atol=0.0)
    assert int(clock.it) == 4
    assert float(clock.elapsed) == pytest.approx(4 * dt)


def test_stage_clocks_read_the_c_i_shift(field_table):
    # du/dt = t; a one-step RK4 integrates exactly to 0.5*dt^2, which
    # requires the stage tendencies to read t + c_i*dt
    dt = 0.4
    stepper = ExplicitRungeKutta(dt, tableaus.RK4)
    modules = (TimeForcing(),)
    schedule = build_schedule(
        field_table, modules, ((0, term_of(TimeForcing.df)),),
        stepper)
    state = make_state(field_table, np.zeros(8))
    _, state, _ = run_eager(stepper, schedule, modules, state, 1)
    assert np.allclose(np.asarray(state["u"].data), 0.5 * dt * dt)


# ================================================================
#  LowStorageRK3 vs ExplicitRungeKutta(RK3) — same order
# ================================================================
def test_low_storage_matches_rk3_order(field_table, u0):
    dt = 0.05
    steps = 20
    schedule_ls, modules_ls = decay_schedule(
        field_table, LowStorageRK3(dt))
    _, ls_state, _ = run_eager(
        LowStorageRK3(dt), schedule_ls, modules_ls,
        make_state(field_table, u0), steps)
    schedule_rk, modules_rk = decay_schedule(
        field_table, ExplicitRungeKutta(dt, tableaus.RK3))
    _, rk_state, _ = run_eager(
        ExplicitRungeKutta(dt, tableaus.RK3), schedule_rk,
        modules_rk, make_state(field_table, u0), steps)
    exact = u0 * np.exp(-LAM * steps * dt)
    ls_err = np.max(np.abs(np.asarray(ls_state["u"].data) - exact))
    rk_err = np.max(np.abs(np.asarray(rk_state["u"].data) - exact))
    # both are third order: comparable error magnitudes, both small
    assert ls_err < 1e-4
    assert rk_err < 1e-4
    assert ls_err == pytest.approx(rk_err, rel=0.1)


# ================================================================
#  Measured convergence order on a smooth problem
# ================================================================
@pytest.mark.parametrize(
    ("tableau", "expected_order"),
    [(tableaus.RK2, 2), (tableaus.RK3, 3), (tableaus.RK4, 4)])
def test_convergence_order(field_table, u0, tableau,
                           expected_order):
    def final_error(dt):
        steps = round(1.0 / dt)
        stepper = ExplicitRungeKutta(dt, tableau)
        schedule, modules = decay_schedule(field_table, stepper)
        _, state, _ = run_eager(
            stepper, schedule, modules,
            make_state(field_table, u0), steps)
        exact = u0 * np.exp(-LAM * 1.0)
        return np.max(np.abs(np.asarray(state["u"].data) - exact))

    coarse = final_error(0.02)
    fine = final_error(0.01)
    order = np.log2(coarse / fine)
    assert order == pytest.approx(expected_order, abs=0.4)


# ================================================================
#  CONSTRAINT placement — project-the-state per stage + final
# ================================================================
@pytest.mark.parametrize(
    ("tableau", "expected_calls"),
    [(tableaus.RK2, 3), (tableaus.RK3, 4), (tableaus.RK4, 5)])
def test_constrain_fires_per_stage_and_final(field_table, u0,
                                             tableau, expected_calls):
    stepper = ExplicitRungeKutta(0.1, tableau)
    counter = Counter()
    modules = (Decay(), counter)
    schedule = build_schedule(
        field_table, modules, ((0, term_of(Decay.du)),), stepper,
        stages=((1, Stage(kind=StageKind.CONSTRAINT,
                          fn=Counter.project, name="project")),))
    run_eager(stepper, schedule, modules,
              make_state(field_table, u0), 1)
    # one per produced stage state (== stage count) plus the final
    # combination substage
    assert counter.calls == expected_calls


def test_low_storage_constrain_fires_per_stage_and_final(
        field_table, u0):
    stepper = LowStorageRK3(0.1)
    counter = Counter()
    modules = (Decay(), counter)
    schedule = build_schedule(
        field_table, modules, ((0, term_of(Decay.du)),), stepper,
        stages=((1, Stage(kind=StageKind.CONSTRAINT,
                          fn=Counter.project, name="project")),))
    run_eager(stepper, schedule, modules,
              make_state(field_table, u0), 1)
    assert counter.calls == 4  # 3 stages + final


# ================================================================
#  jit + lax.scan — one compile
# ================================================================
def test_jit_scan_compiles_once(field_table, u0, compile_counter):
    steps = 8
    stepper = ExplicitRungeKutta(0.05, tableaus.RK4)
    schedule, modules = decay_schedule(field_table, stepper)

    def run(stepper, carry):
        def body(carry, _):
            state, clock = carry
            _, state, clock = stepper.step(
                (), state, schedule.bind(modules), clock)
            return (state, clock), None
        return jax.lax.scan(body, carry, None, length=steps)[0]

    run_jit = jax.jit(run)
    canonical = jax.jit(lambda tree: tree)
    carry = canonical((make_state(field_table, u0), Clock()))
    stepper_in = canonical(stepper)
    compile_counter.reset()
    final_state, _ = run_jit(stepper_in, carry)
    compiles = compile_counter.count
    assert compiles >= 1

    # a dt sweep shares the compile (dt is the only dynamic leaf)
    other = canonical(ExplicitRungeKutta(0.025, tableaus.RK4))
    run_jit(other, canonical((make_state(field_table, u0), Clock())))
    assert compile_counter.count == compiles

    _, eager, _ = run_eager(stepper, schedule, modules,
                            make_state(field_table, u0), steps)
    assert np.allclose(np.asarray(final_state["u"].data),
                       np.asarray(eager["u"].data), rtol=1e-12)


# ================================================================
#  Deferred surfaces
# ================================================================
def test_time_discretization_effect_deferred():
    with pytest.raises(NotImplementedError, match="parity"):
        ExplicitRungeKutta(0.1).time_discretization_effect(
            np.array([1.0]))
    with pytest.raises(NotImplementedError, match="parity"):
        LowStorageRK3(0.1).time_discretization_effect(
            np.array([1.0]))
