"""The IMEX phase loop (model/time_steppers/imex.py).

Prefix-mirrored shard of ``test_imex.py``: the per-phase solve
selection (each merge group solved in the phase owning its fields),
the merged F / state rings, the single tick and warm-up bump, and the
``phases=Phases.total()`` bitwise reduction to the unphased body.
Self-contained per the AGENTS oversized-module rule.
"""
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from fridom.model.clock import Clock
from fridom.model.composer import TendencyComposer
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.phases import Phases
from fridom.model.terms import TendencyTerm, Treatment
from fridom.model.time_steppers.imex import IMEXMultistep
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

#: solve/apply trace of one step (module-level: the fake modules are
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


class Core:
    def du(self, state, _ctx):
        return {"u": state["u"] * 0.0 + 1.0}


class Tracer:
    def db(self, state, _ctx):
        return {"b": state["b"] * 0.0 + 2.0}


class DampingOp:

    """A per-field-independent implicit operator (``-x``)."""

    def __init__(self, fields, key=None):
        self.fields = tuple(fields)
        self._key = key

    def apply(self, _module, state, ctx):
        LOG.append(("apply", tuple(self.fields),
                    None if ctx.phase is None else ctx.phase.index))
        return {name: state[name] * -1.0 for name in self.fields}

    def solve(self, _module, rhs, dt_gamma, ctx):
        LOG.append(("solve", tuple(self.fields),
                    None if ctx.phase is None else ctx.phase.index))
        return {name: rhs[name] / (1.0 + dt_gamma)
                for name in self.fields}

    def merge_key(self):
        return self._key

    def merged_with(self, other):
        return DampingOp(
            self.fields + tuple(f for f in other.fields
                                if f not in self.fields), self._key)

    def restricted_to(self, fields):
        keep = frozenset(fields)
        return DampingOp(
            tuple(f for f in self.fields if f in keep), self._key)


class CoupledOp(DampingOp):

    """The same operator WITHOUT the separability seam."""

    restricted_to = None


class PhasedStepperProbe:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})
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
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
    ))


SPLIT = Phases(("u",), ("b",))


def build(field_table, phases, *, op=None):
    if op is None:
        op = DampingOp(("u", "b"), key="family")
    composer = TendencyComposer(
        field_table=field_table, modules=(Core(), Tracer()),
        terms=(
            (0, TendencyTerm(name="du", fn=Core.du)),
            (1, TendencyTerm(name="db", fn=Tracer.db)),
            (0, TendencyTerm(name="damp",
                             treatment=Treatment.IMPLICIT,
                             implicit=op)),
        ),
        stages=(), time_stepper=PhasedStepperProbe(),
        binding_table=None, phases=phases)
    composer.dry_run()
    return composer.schedule


def zero_state(field_table):
    return VectorField({
        r.name: field_table.grid.create_field(r.space, name=r.name)
        for r in field_table})


def one_step(schedule, field_table, *, scheme="cnab2", dt=0.5,
             stepper_state=None, state=None):
    LOG.clear()
    stepper = IMEXMultistep(dt, scheme=scheme)
    bound = schedule.bind((Core(), Tracer()))
    if state is None:
        state = zero_state(field_table)
    if stepper_state is None:
        stepper_state = stepper.init(
            VectorField({name: state[name]
                         for name in schedule.prognostic}))
    return stepper.step(stepper_state, state, bound, Clock())


# ================================================================
#  Per-phase solves
# ================================================================
def test_each_merge_group_solves_in_its_own_phase(field_table):
    schedule = build(field_table, SPLIT)
    one_step(schedule, field_table)
    assert [(entry[1], entry[2]) for entry in LOG
            if entry[0] == "solve"] == [(("u",), 0), (("b",), 1)]


def test_the_forward_apply_runs_per_phase_too(field_table):
    schedule = build(field_table, SPLIT)
    one_step(schedule, field_table)
    assert [(entry[1], entry[2]) for entry in LOG
            if entry[0] == "apply"] == [(("u",), 0), (("b",), 1)]


def test_sbdf_needs_no_apply(field_table):
    schedule = build(field_table, SPLIT)
    one_step(schedule, field_table, scheme="sbdf2")
    assert not [entry for entry in LOG if entry[0] == "apply"]


def test_a_non_separable_straddling_operator_is_refused(field_table):
    with pytest.raises(AssemblyError, match="atomic under by-variable"):
        build(field_table, SPLIT, op=CoupledOp(("u", "b")))


# ================================================================
#  Rings, clock, warm-up
# ================================================================
def test_the_clock_ticks_once_and_the_counter_bumps_once(
        field_table):
    schedule = build(field_table, SPLIT)
    stepper_state, _state, clock = one_step(
        schedule, field_table, dt=0.5)
    assert int(clock.it) == 1
    assert int(stepper_state.warmup) == 1


def test_the_f_ring_holds_one_merged_full_width_level(field_table):
    schedule = build(field_table, SPLIT)
    stepper_state, _state, _clock = one_step(schedule, field_table)
    level = stepper_state.f_history[0]
    assert level.component_names == ("u", "b")
    assert jnp.allclose(level["u"].data, 1.0)
    assert jnp.allclose(level["b"].data, 2.0)


def test_the_state_ring_holds_the_merged_pre_advance_state(
        field_table):
    schedule = build(field_table, SPLIT)
    state = zero_state(field_table)
    state = state.replace(u=state["u"] + 7.0, b=state["b"] + 9.0)
    stepper_state, _out, _clock = one_step(
        schedule, field_table, scheme="sbdf2", state=state)
    level = stepper_state.x_history[0]
    assert jnp.allclose(level["u"].data, 7.0)
    assert jnp.allclose(level["b"].data, 9.0)


def test_each_phase_writes_only_its_own_keys(field_table):
    schedule = build(field_table, SPLIT)
    _stepper_state, out, _clock = one_step(
        schedule, field_table, dt=0.5)
    # FB-Euler warm-up (gamma=1, apply weight 0): x = (x + dt f)/(1+dt)
    assert jnp.allclose(out["u"].data, 0.5 * 1.0 / 1.5)
    assert jnp.allclose(out["b"].data, 0.5 * 2.0 / 1.5)


# ================================================================
#  The unphased reduction
# ================================================================
@pytest.mark.parametrize("scheme", ["cnab2", "sbdf2"])
def test_total_reduces_to_the_literal_unphased_body(
        field_table, scheme):
    plain = build(field_table, None)
    total = build(field_table, Phases.total())
    a_carry, a_out, _ = one_step(plain, field_table, scheme=scheme)
    b_carry, b_out, _ = one_step(total, field_table, scheme=scheme)
    for name in plain.prognostic:
        assert jnp.array_equal(a_out[name].data, b_out[name].data)
        assert jnp.array_equal(a_carry.f_history[0][name].data,
                               b_carry.f_history[0][name].data)


def test_the_unphased_solve_is_one_merged_group(field_table):
    schedule = build(field_table, None)
    one_step(schedule, field_table)
    assert [(entry[1], entry[2]) for entry in LOG
            if entry[0] == "solve"] == [(("u", "b"), None)]
