"""The phase axis on the schedule (model/schedule.py).

Prefix-mirrored shard of ``test_schedule.py``: ``Schedule.phases`` /
``phased`` / ``phase_entries`` / ``with_phases``, the phase-aware
static token and listing, and the ``phase=`` filter on the four
``BoundSchedule`` groups (incl. the ``per_phase`` contribution mask
and ``implicit_in``). Self-contained per the AGENTS oversized-module
rule (the small builders are duplicated).
"""
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from fridom.model.composer import TendencyComposer
from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.phases import Phases, PhaseView
from fridom.model.schedule import Schedule, ScheduleEntry
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


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
        return {"u": state["u"] * 0.5}

    def joint(self, state, ctx):
        # a per_phase term: writes BOTH groups in one hook
        out = {"u": state["u"] * 0.25, "b": state["b"] * 0.75}
        if ctx.phase is None:
            return out
        return {name: out[name]
                for name in ctx.phase.restrict(("u", "b"))}

    def self_update(self, state, _ctx):
        return {"aux_a": state["aux_a"] + 1.0}

    def project(self, state, _ctx):
        return {"u": state["u"] * 2.0}

    def subcycle(self, state, _ctx):
        return {"ps": state["ps"] + 1.0}


class Tracer:
    def db(self, state, _ctx):
        return {"b": state["b"] * 3.0}


class FakeImplicitOp:
    def __init__(self, fields):
        self.fields = tuple(fields)

    def apply(self, _module, state, _ctx):
        return {name: state[name] * 0.1 for name in self.fields}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        return self


class PhasedStepper:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})
    supports_phases = True


# ================================================================
#  Fixtures / builders
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


@pytest.fixture
def cc(mx):
    return mx.center


@pytest.fixture
def field_table(grid, cc):
    return FakeFieldTable(grid, (
        Record("u", cc, Lifecycle.PROGNOSTIC, 0),
        Record("ps", cc, Lifecycle.PROGNOSTIC, 0),
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
        Record("aux_a", cc, Lifecycle.AUXILIARY, 0),
    ))


def make_state(field_table):
    return VectorField({
        r.name: field_table.grid.create_field(r.space, name=r.name)
        + 1.0
        for r in field_table})


def make_ctx(phase=None):
    view = None if phase is None else PhaseView(*phase)
    return StepContext(params={}, clock=jnp.asarray(0.0),
                       dt=jnp.asarray(1.0), stage_dt=jnp.asarray(1.0),
                       phase=view)


def make_schedule(field_table, *, phases=None, implicit=False):
    """Compose (and dry-run) a two-group schedule."""
    terms = [
        (0, TendencyTerm(name="du", fn=Core.du)),
        (0, TendencyTerm(name="joint", fn=Core.joint,
                         per_phase=True)),
        (1, TendencyTerm(name="db", fn=Tracer.db)),
    ]
    if implicit:
        terms.append((1, TendencyTerm(
            name="mix", treatment=Treatment.IMPLICIT,
            implicit=FakeImplicitOp(("b",)))))
    composer = TendencyComposer(
        field_table=field_table, modules=(Core(), Tracer()),
        terms=tuple(terms),
        stages=(
            (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.self_update,
                      name="self_update")),
            (0, Stage(kind=StageKind.ADVANCE, fn=Core.subcycle,
                      name="subcycle", advances=("ps",))),
            (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.project,
                      name="project")),
        ),
        time_stepper=PhasedStepper(), binding_table=None,
        phases=phases)
    composer.dry_run()
    return composer.schedule


# ================================================================
#  Schedule statics
# ================================================================
def test_an_undeclared_axis_is_the_one_total_group(field_table):
    schedule = make_schedule(field_table)
    assert schedule.phases == (frozenset({"u", "ps", "b"}),)
    assert not schedule.phased


def test_a_declared_axis_installs_the_groups(field_table):
    schedule = make_schedule(
        field_table, phases=Phases(("u", "ps"), ("b",)))
    assert schedule.phased
    assert schedule.phases == (frozenset({"u", "ps"}),
                               frozenset({"b"}))


def test_phase_entries_none_is_the_literal_kind_listing(
        field_table):
    schedule = make_schedule(field_table)
    assert (schedule.phase_entries(None, None)
            == schedule.kind_entries(None))


def test_phase_entries_filters_by_resolved_membership(field_table):
    schedule = make_schedule(
        field_table, phases=Phases(("u", "ps"), ("b",)))
    first = {e.key for e in schedule.phase_entries(None, 0)}
    second = {e.key for e in schedule.phase_entries(None, 1)}
    assert first == {"Core/du", "Core/joint"}
    assert second == {"Tracer/db", "Core/joint"}
    # SELF_UPDATE runs in EVERY phase (active_phases is None)
    for index in (0, 1):
        keys = {e.key for e in schedule.phase_entries(
            StageKind.SELF_UPDATE, index)}
        assert keys == {"Core/self_update"}
    # the ADVANCE claim on ps pins the subcycle to group 0
    assert schedule.phase_entries(StageKind.ADVANCE, 1) == ()


def test_the_static_token_is_untouched_without_an_axis(field_table):
    plain = make_schedule(field_table)
    total = make_schedule(field_table, phases=Phases.total())
    assert plain == total
    assert hash(plain) == hash(total)


def test_the_static_token_changes_with_an_axis(field_table):
    plain = make_schedule(field_table)
    split = make_schedule(field_table,
                          phases=Phases(("u", "ps"), ("b",)))
    assert plain != split


def test_describe_lists_the_groups_and_the_memberships(field_table):
    text = make_schedule(
        field_table,
        phases=Phases(("u", "ps"), ("b",))).describe()
    assert "phases (2 groups)" in text
    assert "phase=every" in text          # the SELF_UPDATE stage
    assert "(per_phase)" in text          # the joint term
    assert "2 phases" in repr(make_schedule(
        field_table, phases=Phases(("u", "ps"), ("b",))))


def test_with_phases_keeps_the_rest_of_the_schedule(field_table):
    schedule = make_schedule(field_table, implicit=True)
    rebuilt = schedule.with_phases(
        schedule.entries, (frozenset({"u", "ps"}), frozenset({"b"})),
        schedule.implicit_groups, schedule.implicit_merged, (1,))
    assert rebuilt.prognostic == schedule.prognostic
    assert rebuilt.implicit_merged == schedule.implicit_merged
    assert rebuilt.implicit_phases == (1,)


def test_schedule_accepts_an_explicit_empty_implicit_phase_row():
    # a schedule built by hand (no composer) defaults the row
    entry = ScheduleEntry(key="M/t", kind=None, slot=0, order=0,
                          index=0, fn=lambda *_: {}, gate=("u",),
                          treatment=Treatment.EXPLICIT)
    schedule = Schedule((entry,), prognostic=("u",))
    assert schedule.implicit_phases == ()


# ================================================================
#  BoundSchedule: the phase= filter
# ================================================================
def test_context_attaches_the_phase_view(field_table):
    schedule = make_schedule(field_table,
                             phases=Phases(("u", "ps"), ("b",)))
    bound = schedule.bind((Core(), Tracer()))
    assert bound.context(0.0, dt=1.0, stage_dt=1.0).phase is None
    view = bound.context(0.0, dt=1.0, stage_dt=1.0, phase=1).phase
    assert view.index == 1
    assert view.fields == frozenset({"b"})


def test_tendency_masks_a_per_phase_term_to_its_group(field_table):
    schedule = make_schedule(field_table,
                             phases=Phases(("u", "ps"), ("b",)))
    bound = schedule.bind((Core(), Tracer()))
    state = make_state(field_table)
    first = bound.tendency(state, make_ctx((0, {"u", "ps"})), phase=0)
    # 0.5*u + 0.25*u = 0.75 ; the joint term's b half is DISCARDED
    assert jnp.allclose(first.explicit["u"].data, 0.75)
    assert jnp.allclose(first.explicit["b"].data, 0.0)
    second = bound.tendency(state, make_ctx((1, {"b"})), phase=1)
    assert jnp.allclose(second.explicit["u"].data, 0.0)
    assert jnp.allclose(second.explicit["b"].data, 3.75)  # 3 + 0.75


def test_tendency_keeps_the_full_width_template(field_table):
    schedule = make_schedule(field_table,
                             phases=Phases(("u", "ps"), ("b",)))
    bound = schedule.bind((Core(), Tracer()))
    sums = bound.tendency(make_state(field_table),
                          make_ctx((1, {"b"})), phase=1)
    assert sums.explicit.component_names == ("u", "ps", "b")


def test_unphased_tendency_sums_every_term(field_table):
    schedule = make_schedule(field_table)
    bound = schedule.bind((Core(), Tracer()))
    sums = bound.tendency(make_state(field_table), make_ctx())
    assert jnp.allclose(sums.explicit["u"].data, 0.75)
    assert jnp.allclose(sums.explicit["b"].data, 3.75)


def test_stage_groups_honour_the_phase_filter(field_table):
    schedule = make_schedule(field_table,
                             phases=Phases(("u", "ps"), ("b",)))
    bound = schedule.bind((Core(), Tracer()))
    state = make_state(field_table)
    # SELF_UPDATE runs in both phases; ADVANCE/CONSTRAINT only in 0
    assert jnp.allclose(
        bound.prepare(state, make_ctx((1, {"b"})),
                      phase=1)["aux_a"].data, 2.0)
    assert jnp.allclose(
        bound.advance_stages(state, make_ctx((1, {"b"})),
                             phase=1)["ps"].data, 1.0)
    assert jnp.allclose(
        bound.advance_stages(state, make_ctx((0, {"u", "ps"})),
                             phase=0)["ps"].data, 2.0)
    assert jnp.allclose(
        bound.constrain(state, make_ctx((1, {"b"})),
                        phase=1)["u"].data, 1.0)
    assert jnp.allclose(
        bound.constrain(state, make_ctx((0, {"u", "ps"})),
                        phase=0)["u"].data, 2.0)


def test_implicit_in_selects_the_owning_phase(field_table):
    schedule = make_schedule(field_table, implicit=True,
                             phases=Phases(("u", "ps"), ("b",)))
    bound = schedule.bind((Core(), Tracer()))
    assert schedule.implicit_phases == (1,)
    assert len(bound.implicit_in(None)) == 1
    assert bound.implicit_in(0) == ()
    assert len(bound.implicit_in(1)) == 1
