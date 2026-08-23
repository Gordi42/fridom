"""The AB phase loop (model/time_steppers/adam_bashforth.py).

Prefix-mirrored shard of ``test_adam_bashforth.py``: the ORDER GATE
of the staggered step — the recorded hook sequence per phase, the
full-state read rule across phases, the ``per_phase`` term's two
evaluations, the single clock tick and warm-up bump, and the one
merged full-width ring level. Self-contained per the AGENTS
oversized-module rule.
"""
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from fridom.model.clock import Clock
from fridom.model.composer import TendencyComposer
from fridom.model.declarations import Lifecycle
from fridom.model.phases import Phases
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

#: the hook trace of one step (module-level: the fake modules are
#: rebuilt per step and must share one recorder)
LOG: list = []


# ================================================================
#  Fakes
# ================================================================
class Record(NamedTuple):
    name: str
    space: object
    lifecycle: Lifecycle
    owner: int
    roles: tuple = ()


class FakeFieldTable:
    def __init__(self, grid, records):
        self.grid = grid
        self._records = tuple(records)

    def __iter__(self):
        return iter(self._records)


def _mean(field):
    # record the FIELD, never a host float: the composer's dry run
    # evaluates every hook under jax.eval_shape, where a float() would
    # raise ConcretizationTypeError
    return field


def mean_of(entry):
    return float(jnp.mean(entry[2].data))


class Core:

    """Momentum side: one u term, the ps subcycle, the projection."""

    def du(self, state, ctx):
        LOG.append(("term_u", _phase(ctx), _mean(state["u"])))
        return {"u": state["u"] * 0.0 + 1.0}

    def joint(self, state, ctx):
        LOG.append(("term_joint", _phase(ctx),
                    None if ctx.phase is None
                    else tuple(sorted(ctx.phase.fields))))
        out = {"u": state["u"] * 0.0 + 0.5,
               "b": state["b"] * 0.0 + 0.25}
        if ctx.phase is None:
            return out
        return {name: out[name]
                for name in ctx.phase.restrict(("u", "b"))}

    def refresh(self, state, ctx):
        LOG.append(("self_update", _phase(ctx)))
        return {"aux_a": state["aux_a"] + 1.0}

    def diagnose(self, state, ctx):
        LOG.append(("diagnose", _phase(ctx)))
        return {"diag_a": state["aux_a"] * 1.0}

    def subcycle(self, state, ctx):
        LOG.append(("advance_ps", _phase(ctx)))
        return {"ps": state["ps"] + 10.0}

    def project(self, state, ctx):
        LOG.append(("constrain_u", _phase(ctx), _mean(state["u"])))
        return {"u": state["u"] * 2.0}


class Tracer:

    """Tracer side: the b term and a b-only clamp."""

    def db(self, state, ctx):
        LOG.append(("term_b", _phase(ctx), _mean(state["u"])))
        return {"b": state["b"] * 0.0 + 3.0}

    def clamp(self, state, ctx):
        LOG.append(("constrain_b", _phase(ctx)))
        return {"b": state["b"] * 1.0}


def _phase(ctx):
    return None if ctx.phase is None else ctx.phase.index


class PhasedStepperProbe:
    supported_treatments = frozenset({Treatment.EXPLICIT})
    supports_phases = True


# ================================================================
#  Builders
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(4, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


@pytest.fixture
def field_table(grid, mx):
    cc = mx.center
    return FakeFieldTable(grid, (
        Record("u", cc, Lifecycle.PROGNOSTIC, 0),
        Record("ps", cc, Lifecycle.PROGNOSTIC, 0),
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
        Record("aux_a", cc, Lifecycle.AUXILIARY, 0),
        Record("diag_a", cc, Lifecycle.DIAGNOSTIC, 0),
    ))


def build(field_table, phases):
    composer = TendencyComposer(
        field_table=field_table, modules=(Core(), Tracer()),
        terms=(
            (0, TendencyTerm(name="du", fn=Core.du)),
            (0, TendencyTerm(name="joint", fn=Core.joint,
                             per_phase=True)),
            (1, TendencyTerm(name="db", fn=Tracer.db)),
        ),
        stages=(
            (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.refresh,
                      name="refresh")),
            (0, Stage(kind=StageKind.DIAGNOSE, fn=Core.diagnose,
                      name="diagnose")),
            (0, Stage(kind=StageKind.ADVANCE, fn=Core.subcycle,
                      name="subcycle", advances=("ps",))),
            (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.project,
                      name="project")),
            (1, Stage(kind=StageKind.CONSTRAINT, fn=Tracer.clamp,
                      name="clamp", order=1)),
        ),
        time_stepper=PhasedStepperProbe(), binding_table=None,
        phases=phases)
    composer.dry_run()
    return composer.schedule


def zero_state(field_table):
    return VectorField({
        r.name: field_table.grid.create_field(r.space, name=r.name)
        for r in field_table})


def one_step(schedule, field_table, *, order=2, dt=0.5,
             stepper_state=None, state=None):
    LOG.clear()
    stepper = AdamBashforth(dt, order=order)
    bound = schedule.bind((Core(), Tracer()))
    if state is None:
        state = zero_state(field_table)
    if stepper_state is None:
        stepper_state = stepper.init(
            VectorField({name: state[name]
                         for name in schedule.prognostic}))
    return stepper.step(stepper_state, state, bound, Clock())


# ================================================================
#  The order gate
# ================================================================
def test_two_phases_run_the_canonical_chain_each(field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    one_step(schedule, field_table)
    assert [entry[0] for entry in LOG] == [
        # phase 0: S1 S1' S2 (S3) S3' S4
        "self_update", "diagnose", "term_u", "term_joint",
        "advance_ps", "constrain_u",
        # phase 1: S1 S1' S2 (S3) S4 — no ADVANCE claims a tracer
        "self_update", "diagnose", "term_joint", "term_b",
        "constrain_b",
    ]
    assert [entry[1] for entry in LOG] == [0] * 6 + [1] * 5


def test_the_unphased_chain_is_the_single_pass(field_table):
    schedule = build(field_table, None)
    one_step(schedule, field_table)
    assert [entry[0] for entry in LOG] == [
        "self_update", "diagnose", "term_u", "term_joint", "term_b",
        "advance_ps", "constrain_u", "constrain_b"]
    assert all(entry[1] is None for entry in LOG)


def test_a_per_phase_term_runs_once_per_phase_with_its_fields(
        field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    one_step(schedule, field_table)
    seen = [entry[2] for entry in LOG if entry[0] == "term_joint"]
    assert seen == [("ps", "u"), ("b",)]


def test_phase_one_reads_the_phase_zero_advanced_velocity(
        field_table):
    # the full-state read rule ACROSS phases: the tracer term sees
    # u after phase 0's advance AND its projection (0 -> dt*1.5 -> x2)
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    _stepper_state, _state, _clock = one_step(
        schedule, field_table, dt=0.5)
    read_first = mean_of(next(e for e in LOG if e[0] == "term_u"))
    read_last = mean_of(next(e for e in LOG if e[0] == "term_b"))
    assert read_first == 0.0
    # AB1 warm-up row: u += dt * (1.0 + 0.5) = 0.75, then x2
    assert read_last == pytest.approx(1.5)


def test_the_clock_ticks_once_for_the_whole_step(field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    _stepper_state, _state, clock = one_step(
        schedule, field_table, dt=0.5)
    assert float(clock.time) == pytest.approx(0.5)
    assert int(clock.it) == 1


def test_the_warmup_counter_bumps_once(field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    stepper_state, _state, _clock = one_step(
        schedule, field_table, order=3)
    assert int(stepper_state.warmup) == 1


def test_the_ring_holds_one_merged_full_width_level(field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    stepper_state, _state, _clock = one_step(schedule, field_table)
    level = stepper_state.history[0]
    assert level.component_names == ("u", "ps", "b")
    # u from phase 0 (1.0 + 0.5), b from phase 1 (3.0 + 0.25)
    assert jnp.allclose(level["u"].data, 1.5)
    assert jnp.allclose(level["b"].data, 3.25)
    assert jnp.allclose(level["ps"].data, 0.0)


def test_only_the_phases_own_keys_are_advanced_per_phase(
        field_table):
    # b must NOT pick up phase 0's (zero) b-row increment twice: the
    # state.add per phase is restricted to that phase's keys
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    _stepper_state, state, _clock = one_step(
        schedule, field_table, dt=0.5)
    assert jnp.allclose(state["b"].data, 0.5 * 3.25)
    assert jnp.allclose(state["u"].data, 2.0 * 0.5 * 1.5)
    assert jnp.allclose(state["ps"].data, 10.0)


def test_a_second_step_consumes_the_merged_history(field_table):
    schedule = build(field_table, Phases(("u", "ps"), ("b",)))
    stepper_state, state, _clock = one_step(
        schedule, field_table, dt=0.5)
    stepper_state, state, _clock = one_step(
        schedule, field_table, dt=0.5,
        stepper_state=stepper_state, state=state)
    # AB2 row [3/2, -1/2] on identical levels -> dt * value
    level = stepper_state.history[0]
    assert jnp.allclose(level["b"].data, 3.25)
    assert int(stepper_state.warmup) == 1


def test_a_single_group_takes_the_literal_unphased_body(
        field_table):
    plain = build(field_table, None)
    total = build(field_table, Phases.total())
    a_state, a_out, _ = one_step(plain, field_table)
    b_state, b_out, _ = one_step(total, field_table)
    for name in plain.prognostic:
        assert jnp.array_equal(a_out[name].data, b_out[name].data)
        assert jnp.array_equal(a_state.history[0][name].data,
                               b_state.history[0][name].data)
