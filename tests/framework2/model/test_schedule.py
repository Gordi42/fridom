"""Tests for the static schedule (framework2/model/schedule.py).

Covers Schedule equality/hash across identical configurations, the
BoundSchedule stage groups over real state (incl. SELF_UPDATE-first
observed by a DIAGNOSE reader), the context construction seam, the
wave-5 NotImplementedError stubs, and the TendencySums pytree.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.model.composer import TendencyComposer
from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.schedule import (
    Schedule,
    ScheduleEntry,
    TendencySums,
)
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment


# ================================================================
#  Fakes (duck-typed composer inputs)
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


class Core:
    def du(self, state, _ctx):
        return {"u": state["b"] * 0.5}

    def self_update(self, state, _ctx):
        return {"aux_a": state["aux_a"] + 5.0}

    def diagnose(self, state, _ctx):
        # reads the AUX field the self_update just rewrote
        return {"diag_a": state["aux_a"] * 1.0}

    def project(self, state, _ctx):
        return {"u": state["u"] * 0.5}

    def accumulate(self, state, _ctx):
        return {"diag_a": state["diag_a"] + 1.0}


class Forcing:
    def db(self, state, _ctx):
        return {"b": state["u"] * 0.0 + 2.0}


class FakeImplicitOp:
    def __init__(self, fields):
        self.fields = tuple(fields)

    def apply(self, _module, state, _ctx):
        return {name: state[name] * 100.0 for name in self.fields}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        return self


class FakeBindingTable:
    def __init__(self):
        self.calls = []

    def eval_params(self, modules, stepper, t):
        self.calls.append((modules, stepper, t))
        return {"scaling.ro": jnp.asarray(2.0)}


class FakeStepper:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 1.0), name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def cc(mx, my):
    return mx.center * my.center


@pytest.fixture
def field_table(grid, cc):
    return FakeFieldTable(grid, (
        Record("u", cc, Lifecycle.PROGNOSTIC, 0),
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
        Record("aux_a", cc, Lifecycle.AUXILIARY, 0),
        Record("diag_a", cc, Lifecycle.DIAGNOSTIC, 0),
    ))


def default_stages():
    return (
        (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.self_update,
                  name="self_update")),
        (0, Stage(kind=StageKind.DIAGNOSE, fn=Core.diagnose,
                  name="diagnose")),
        (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.project,
                  name="project")),
        (0, Stage(kind=StageKind.DIAGNOSTIC, fn=Core.accumulate,
                  name="accumulate")),
    )


def make_composer(field_table, *, binding_table=None,
                  time_stepper=None, stages=None, order_tweak=0):
    if stages is None:
        stages = default_stages()
    if order_tweak:
        slot, stage = stages[-1]
        stages = (*stages[:-1],
                  (slot, Stage(kind=stage.kind, fn=stage.fn,
                               name=stage.name, order=order_tweak)))
    return TendencyComposer(
        field_table=field_table,
        modules=(Core(), Forcing()),
        terms=(
            (0, TendencyTerm(name="du", fn=Core.du)),
            (1, TendencyTerm(name="db", fn=Forcing.db)),
        ),
        stages=stages,
        time_stepper=time_stepper,
        binding_table=binding_table,
    )


def make_state(field_table):
    return VectorField({
        r.name: field_table.grid.create_field(r.space, name=r.name)
        for r in field_table})


def make_ctx(sums=None):
    return StepContext(params={}, clock=jnp.asarray(0.0),
                       dt=jnp.asarray(1.0),
                       stage_dt=jnp.asarray(1.0),
                       tendency_sums=sums)


# ================================================================
#  Schedule statics: equality, hash, ordering
# ================================================================
def test_identical_configurations_produce_equal_schedules(
        field_table):
    s1 = make_composer(field_table).schedule
    s2 = make_composer(field_table).schedule
    assert s1 is not s2
    assert s1 == s2
    assert hash(s1) == hash(s2)


def test_different_order_produces_unequal_schedules(field_table):
    s1 = make_composer(field_table).schedule
    s2 = make_composer(field_table, order_tweak=3).schedule
    assert s1 != s2


def test_schedule_is_hash_stable_and_usable_as_key(field_table):
    s1 = make_composer(field_table).schedule
    s2 = make_composer(field_table).schedule
    cache = {s1: "compiled"}
    assert cache[s2] == "compiled"  # the shared-jit-cache obligation


def test_entries_are_kind_ordered(field_table):
    schedule = make_composer(field_table).schedule
    ranks = [entry.rank for entry in schedule.entries]
    assert ranks == sorted(ranks)
    assert schedule.entries[0].kind is StageKind.SELF_UPDATE


def test_prognostic_names_in_declaration_order(field_table):
    schedule = make_composer(field_table).schedule
    assert schedule.prognostic == ("u", "b")


def test_describe_lists_attribution_keys(field_table):
    text = make_composer(field_table).schedule.describe()
    assert "Core/du" in text
    assert "Forcing/db" in text
    assert "SELF_UPDATE" in text
    assert "CONSTRAINT" in text


# ================================================================
#  bind() and the stage groups over real state
# ================================================================
def test_bind_checks_module_tuple_length(field_table):
    schedule = make_composer(field_table).schedule
    with pytest.raises(AssemblyError, match="slot"):
        schedule.bind((Core(),))


def test_prepare_runs_self_update_before_diagnose(field_table):
    # the load-bearing SELF_UPDATE-first pin: the DIAGNOSE reader
    # must observe the owner's fresh AUX write from the same substage
    modules = (Core(), Forcing())
    bound = make_composer(field_table).schedule.bind(modules)
    state = make_state(field_table)
    assert jnp.allclose(state["aux_a"].data, 0.0)
    prepared = bound.prepare(state, make_ctx())
    assert jnp.allclose(prepared["aux_a"].data, 5.0)
    assert jnp.allclose(prepared["diag_a"].data, 5.0)


def test_tendency_accumulates_explicit_sums(field_table):
    modules = (Core(), Forcing())
    bound = make_composer(field_table).schedule.bind(modules)
    state = make_state(field_table).replace(
        b=make_state(field_table)["b"] + 4.0)
    sums = bound.tendency(state, make_ctx())
    assert isinstance(sums, TendencySums)
    assert sums.explicit.component_names == ("u", "b")
    assert jnp.allclose(sums.explicit["u"].data, 2.0)  # b * 0.5
    assert jnp.allclose(sums.explicit["b"].data, 2.0)  # constant
    # the input state is untouched (functional accumulation)
    assert jnp.allclose(state["u"].data, 0.0)


def test_tendency_skips_implicit_terms(field_table):
    op = FakeImplicitOp(fields=("u",))
    composer = TendencyComposer(
        field_table=field_table,
        modules=(Core(), Forcing()),
        terms=(
            (0, TendencyTerm(name="du", fn=Core.du)),
            (1, TendencyTerm(name="db", fn=Forcing.db)),
            (0, TendencyTerm(name="mix",
                             treatment=Treatment.IMPLICIT,
                             implicit=op)),
        ),
        stages=(),
        time_stepper=FakeStepper(),
        binding_table=None,
    )
    state = make_state(field_table).replace(
        b=make_state(field_table)["b"] + 4.0)
    sums = composer.schedule.bind((Core(), Forcing())).tendency(
        state, make_ctx())
    # the op's apply (x * 100) must NOT appear in the explicit sums
    assert jnp.allclose(sums.explicit["u"].data, 2.0)


def test_constrain_applies_constraint_stages(field_table):
    modules = (Core(), Forcing())
    bound = make_composer(field_table).schedule.bind(modules)
    state = make_state(field_table).replace(
        u=make_state(field_table)["u"] + 8.0)
    constrained = bound.constrain(state, make_ctx())
    assert jnp.allclose(constrained["u"].data, 4.0)


def test_diagnostics_supports_the_accumulation_idiom(field_table):
    modules = (Core(), Forcing())
    bound = make_composer(field_table).schedule.bind(modules)
    state = make_state(field_table)
    state = bound.diagnostics(state, make_ctx())
    state = bound.diagnostics(state, make_ctx())
    assert jnp.allclose(state["diag_a"].data, 2.0)


def test_stage_groups_work_under_jit(field_table):
    schedule = make_composer(field_table).schedule
    modules = (Core(), Forcing())

    @jax.jit
    def run(state, ctx):
        bound = schedule.bind(modules)
        state = bound.prepare(state, ctx)
        sums = bound.tendency(state, ctx)
        return bound.constrain(state, ctx), sums

    state = make_state(field_table)
    out, sums = run(state, make_ctx())
    assert jnp.allclose(out["aux_a"].data, 5.0)
    assert jnp.allclose(sums.explicit["b"].data, 2.0)


# ================================================================
#  Wave-5 seams (landed 2.5)
# ================================================================
def test_advance_stages_runs_the_advance_group(field_table):
    # the default composition declares no ADVANCE stages, so the
    # group is an identity pass-through (the sequential replace walk)
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    state = make_state(field_table).replace(
        u=make_state(field_table)["u"] + 3.0)
    out = bound.advance_stages(state, make_ctx())
    assert jnp.allclose(out["u"].data, 3.0)


def test_implicit_is_empty_without_implicit_terms(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    assert bound.implicit == ()


def test_tendency_sums_getitem_by_treatment(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    sums = bound.tendency(make_state(field_table), make_ctx())
    assert sums[Treatment.EXPLICIT] is sums.explicit
    # the IMPLICIT entry is absent unless the scheme computed the
    # forward applies this step (V-H4)
    with pytest.raises(KeyError):
        _ = sums[Treatment.IMPLICIT]


# ================================================================
#  context() — the P0 construction seam
# ================================================================
def test_context_without_binding_table_has_empty_params(
        field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    ctx = bound.context(jnp.asarray(3.0), dt=60.0, stage_dt=30.0)
    assert isinstance(ctx, StepContext)
    assert dict(ctx.params) == {}
    assert ctx.clock == pytest.approx(3.0)
    assert ctx.dt == pytest.approx(60.0)
    assert ctx.stage_dt == pytest.approx(30.0)
    assert ctx.tendency_sums is None


def test_context_reads_params_through_the_binding_table(
        field_table):
    table = FakeBindingTable()
    stepper = FakeStepper()
    composer = make_composer(field_table, binding_table=table,
                             time_stepper=stepper)
    modules = (Core(), Forcing())
    bound = composer.schedule.bind(modules)
    ctx = bound.context(jnp.asarray(7.0), dt=1.0, stage_dt=1.0)
    assert ctx.params["scaling.ro"] == pytest.approx(2.0)
    (called_modules, called_stepper, called_t) = table.calls[0]
    assert called_modules == modules
    assert called_stepper is stepper
    assert called_t == pytest.approx(7.0)


def test_context_attaches_tendency_sums(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    sums = bound.tendency(make_state(field_table), make_ctx())
    ctx = bound.context(jnp.asarray(0.0), dt=1.0, stage_dt=1.0,
                        sums=sums)
    assert ctx.tendency_sums is sums


# ================================================================
#  TendencySums — frozen jaxified pytree
# ================================================================
def test_tendency_sums_jaxify_round_trip(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    state = make_state(field_table).replace(
        b=make_state(field_table)["b"] + 4.0)
    sums = bound.tendency(state, make_ctx())
    leaves, treedef = jax.tree_util.tree_flatten(sums)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is TendencySums
    assert rebuilt.explicit.component_names == ("u", "b")
    assert jnp.array_equal(rebuilt.explicit["u"].data,
                           sums.explicit["u"].data)


def test_tendency_sums_is_frozen(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    sums = bound.tendency(make_state(field_table), make_ctx())
    with pytest.raises(AttributeError, match="frozen"):
        sums.explicit = None
    with pytest.raises(AttributeError, match="frozen"):
        del sums.explicit


def test_tendency_sums_usable_inside_jit(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    sums = bound.tendency(make_state(field_table), make_ctx())

    @jax.jit
    def scale(s):
        return s.explicit["b"].data * 2.0

    assert jnp.allclose(scale(sums), 4.0)


def test_tendency_sums_repr_smoke(field_table):
    bound = make_composer(field_table).schedule.bind(
        (Core(), Forcing()))
    sums = bound.tendency(make_state(field_table), make_ctx())
    assert "TendencySums" in repr(sums)


# ================================================================
#  Schedule construction sanity
# ================================================================
def test_schedule_repr_smoke(field_table):
    assert "Schedule(" in repr(make_composer(field_table).schedule)


def test_manual_schedule_equality_ignores_fn_identity():
    # the statics token is attribution keys/kinds/orders — the
    # unbound callables (identity-hashed) are deliberately excluded
    def make(fn):
        entry = ScheduleEntry(
            key="M/t", kind=None, slot=0, order=0, index=0,
            fn=fn, gate=("u",), treatment=Treatment.EXPLICIT)
        return Schedule((entry,), prognostic=("u",))

    s1 = make(lambda *_args: {})
    s2 = make(lambda *_args: {})
    assert s1 == s2
    assert hash(s1) == hash(s2)
