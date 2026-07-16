"""Tests for the tendency composer (model/composer.py).

Covers the wave-3 composer surface: deterministic accumulation
order, the write-gate matrix per kind, the same-kind overlap lint,
attribution in TermEvaluationError, the dry-run validation over real
zero-valued fields, coverage/advances cross-checks, implicit merge
grouping, and composed-body purity (incl. under jax.jit).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from fridom.model.composer import TendencyComposer
from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.errors import (
    AssemblyError,
    ImplicitCollisionError,
    TermEvaluationError,
)
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


# ================================================================
#  Fakes (duck-typed composer inputs; no Module/FieldTable imports)
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

    """Fake dynamical-core module (slot 0 by convention)."""

    def du(self, state, _ctx):
        return {"u": state["b"] * 0.5}

    def update_aux(self, state, _ctx):
        return {"aux_a": state["aux_a"] + 1.0}

    def diagnose(self, state, _ctx):
        return {"diag_a": state["aux_a"] * 2.0}


class Forcing:

    """Fake forcing module (slot 1 by convention)."""

    def db(self, state, _ctx):
        return {"b": state["u"] * 0.25 + 1.0}


class FakeStepper:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})


class ExplicitOnlyStepper:
    supported_treatments = frozenset({Treatment.EXPLICIT})


class FakeImplicitOp:
    def __init__(self, fields, key=None):
        self.fields = tuple(fields)
        self._key = key

    def apply(self, _module, state, _ctx):
        return {name: state[name] * 0.1 for name in self.fields}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return self._key

    def merged_with(self, _other):
        return self


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
        Record("diag_b", cc, Lifecycle.DIAGNOSTIC, 1),
    ))


def make_composer(field_table, *, modules=None, terms=None,
                  stages=(), time_stepper=None, binding_table=None,
                  term_filter=None):
    if modules is None:
        modules = (Core(), Forcing())
    if terms is None:
        terms = (
            (0, TendencyTerm(name="du", fn=Core.du)),
            (1, TendencyTerm(name="db", fn=Forcing.db)),
        )
    return TendencyComposer(
        field_table=field_table, modules=modules, terms=terms,
        stages=stages, time_stepper=time_stepper,
        binding_table=binding_table, term_filter=term_filter)


def make_ctx():
    return StepContext(params={}, clock=jnp.asarray(0.0),
                       dt=jnp.asarray(1.0), stage_dt=jnp.asarray(1.0))


def _full_state(field_table):
    return VectorField({
        r.name: field_table.grid.create_field(r.space, name=r.name)
        for r in field_table})


# ================================================================
#  Deterministic accumulation order
# ================================================================
def test_accumulation_order_is_module_then_declaration(field_table):
    calls = []

    class A:
        def t1(self, _state, _ctx):
            calls.append("A/t1")
            return {}

        def t2(self, _state, _ctx):
            calls.append("A/t2")
            return {}

    class B:
        def t1(self, _state, _ctx):
            calls.append("B/t1")
            return {}

        def t2(self, _state, _ctx):
            calls.append("B/t2")
            return {}

    # collection order deliberately interleaved: the DEFINED order
    # is module tuple order, then declaration order within a module
    terms = (
        (1, TendencyTerm(name="t1", fn=B.t1)),
        (0, TendencyTerm(name="t1", fn=A.t1)),
        (1, TendencyTerm(name="t2", fn=B.t2)),
        (0, TendencyTerm(name="t2", fn=A.t2)),
    )
    composer = make_composer(field_table, modules=(A(), B()),
                             terms=terms)
    composer.schedule.bind((A(), B())).tendency(
        _full_state(field_table), make_ctx())
    assert calls == ["A/t1", "A/t2", "B/t1", "B/t2"]

    # permuting the module tuple permutes the defined order
    calls.clear()
    terms_swapped = (
        (0, TendencyTerm(name="t1", fn=B.t1)),
        (1, TendencyTerm(name="t1", fn=A.t1)),
        (0, TendencyTerm(name="t2", fn=B.t2)),
        (1, TendencyTerm(name="t2", fn=A.t2)),
    )
    composer = make_composer(field_table, modules=(B(), A()),
                             terms=terms_swapped)
    composer.schedule.bind((B(), A())).tendency(
        _full_state(field_table), make_ctx())
    assert calls == ["B/t1", "B/t2", "A/t1", "A/t2"]


# ================================================================
#  Write-gate matrix (positive + negative per kind)
# ================================================================
def test_term_gate_prognostic_ok(field_table):
    make_composer(field_table).dry_run()  # writes u and b


def test_term_gate_rejects_non_prognostic(field_table):
    class Bad:
        def t(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

    terms = (
        (0, TendencyTerm(name="du", fn=Core.du)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
        (0, TendencyTerm(name="t", fn=Bad.t)),
    )
    composer = make_composer(field_table, modules=(Bad(), Forcing()),
                             terms=terms)
    with pytest.raises(AssemblyError, match=r"Bad/t.*write gate"):
        composer.dry_run()


def test_dry_run_catches_bad_contribution_key(field_table):
    class Typo:
        def t(self, state, _ctx):
            return {"uu": state["u"] * 1.0}  # typo'd key

    terms = (
        (0, TendencyTerm(name="du", fn=Core.du)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
        (0, TendencyTerm(name="t", fn=Typo.t)),
    )
    composer = make_composer(field_table,
                             modules=(Typo(), Forcing()),
                             terms=terms)
    with pytest.raises(AssemblyError, match=r"Typo/t.*'uu'"):
        composer.dry_run()


def test_self_update_gate_own_aux(field_table):
    stage = Stage(kind=StageKind.SELF_UPDATE, fn=Core.update_aux,
                  name="update_aux")
    composer = make_composer(field_table, stages=((0, stage),))
    composer.dry_run()


def test_self_update_gate_rejects_prognostic(field_table):
    class Rogue:
        def up(self, state, _ctx):
            return {"u": state["u"] + 1.0}

    stage = Stage(kind=StageKind.SELF_UPDATE, fn=Rogue.up, name="up")
    composer = make_composer(
        field_table, modules=(Rogue(), Forcing()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=((0, stage),))
    with pytest.raises(AssemblyError, match=r"Rogue/up.*write gate"):
        composer.dry_run()


def test_self_update_gate_rejects_foreign_aux(field_table):
    # aux_a is owned by slot 0; a slot-1 SELF_UPDATE may not write it
    class Intruder:
        def up(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

    stage = Stage(kind=StageKind.SELF_UPDATE, fn=Intruder.up,
                  name="up")
    composer = make_composer(
        field_table, modules=(Core(), Intruder()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=((1, stage),))
    with pytest.raises(AssemblyError,
                       match=r"Intruder/up.*write gate"):
        composer.dry_run()


def test_diagnose_gate_own_diagnostic(field_table):
    stage = Stage(kind=StageKind.DIAGNOSE, fn=Core.diagnose,
                  name="diagnose")
    composer = make_composer(field_table, stages=((0, stage),))
    composer.dry_run()


def test_diagnose_gate_rejects_aux(field_table):
    class BadDiag:
        def d(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

    stage = Stage(kind=StageKind.DIAGNOSE, fn=BadDiag.d, name="d")
    composer = make_composer(
        field_table, modules=(BadDiag(), Forcing()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=((0, stage),))
    with pytest.raises(AssemblyError,
                       match=r"BadDiag/d.*write gate"):
        composer.dry_run()


def test_advance_gate_declared_advances_plus_own_aux(field_table):
    class Subcycle:
        def adv(self, state, _ctx):
            return {"u": state["u"] + 1.0,
                    "aux_a": state["aux_a"] + 1.0}

    stage = Stage(kind=StageKind.ADVANCE, fn=Subcycle.adv,
                  name="adv", advances=("u",))
    composer = make_composer(
        field_table, modules=(Subcycle(), Forcing()),
        terms=((1, TendencyTerm(name="db", fn=Forcing.db)),),
        stages=((0, stage),))
    composer.dry_run()


def test_advance_gate_rejects_undeclared_prognostic(field_table):
    class Subcycle:
        def adv(self, state, _ctx):
            return {"b": state["b"] + 1.0}  # only "u" declared

    stage = Stage(kind=StageKind.ADVANCE, fn=Subcycle.adv,
                  name="adv", advances=("u",))
    composer = make_composer(
        field_table, modules=(Subcycle(), Forcing()),
        stages=((0, stage),))
    with pytest.raises(AssemblyError,
                       match=r"Subcycle/adv.*write gate"):
        composer.dry_run()


def test_constraint_gate_prognostic_and_own_diag(field_table):
    class Projection:
        def project(self, state, _ctx):
            return {"u": state["u"] * 0.5,
                    "diag_a": state["u"] * 1.0}

    stage = Stage(kind=StageKind.CONSTRAINT, fn=Projection.project,
                  name="project")
    composer = make_composer(
        field_table, modules=(Projection(), Forcing()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=((0, stage),))
    composer.dry_run()


def test_constraint_gate_rejects_aux(field_table):
    class BadProj:
        def project(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

    stage = Stage(kind=StageKind.CONSTRAINT, fn=BadProj.project,
                  name="project")
    composer = make_composer(
        field_table, modules=(BadProj(), Forcing()),
        stages=((0, stage),))
    with pytest.raises(AssemblyError,
                       match=r"BadProj/project.*write gate"):
        composer.dry_run()


def test_diagnostic_gate_rejects_foreign_diag(field_table):
    # diag_b is owned by slot 1; a slot-0 DIAGNOSTIC may not write it
    class Accum:
        def acc(self, state, _ctx):
            return {"diag_b": state["diag_b"] + 1.0}

    stage = Stage(kind=StageKind.DIAGNOSTIC, fn=Accum.acc,
                  name="acc")
    composer = make_composer(
        field_table, modules=(Accum(), Forcing()),
        stages=((0, stage),))
    with pytest.raises(AssemblyError,
                       match=r"Accum/acc.*write gate"):
        composer.dry_run()


def test_diagnostic_gate_own_diag_accumulation_idiom(field_table):
    class Accum:
        def acc(self, state, _ctx):
            # reads its own previous value — the accumulation idiom
            return {"diag_a": state["diag_a"] + 1.0}

    stage = Stage(kind=StageKind.DIAGNOSTIC, fn=Accum.acc,
                  name="acc")
    composer = make_composer(
        field_table, modules=(Accum(), Forcing()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=((0, stage),))
    composer.dry_run()


# ================================================================
#  Same-kind overlap lint
# ================================================================
def test_same_kind_equal_order_overlap_errors(field_table):
    class P1:
        def proj(self, state, _ctx):
            return {"u": state["u"] * 0.5}

    class P2:
        def proj(self, state, _ctx):
            return {"u": state["u"] * 0.25}

    stages = (
        (0, Stage(kind=StageKind.CONSTRAINT, fn=P1.proj,
                  name="proj")),
        (1, Stage(kind=StageKind.CONSTRAINT, fn=P2.proj,
                  name="proj")),
    )
    composer = make_composer(
        field_table, modules=(P1(), P2()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=stages)
    with pytest.raises(AssemblyError, match="explicit order="):
        composer.dry_run()


def test_same_kind_distinct_order_overlap_passes(field_table):
    class P1:
        def proj(self, state, _ctx):
            return {"u": state["u"] * 0.5}

    class P2:
        def proj(self, state, _ctx):
            return {"u": state["u"] * 0.25}

    stages = (
        (0, Stage(kind=StageKind.CONSTRAINT, fn=P1.proj,
                  name="proj", order=0)),
        (1, Stage(kind=StageKind.CONSTRAINT, fn=P2.proj,
                  name="proj", order=1)),
    )
    composer = make_composer(
        field_table, modules=(P1(), P2()),
        terms=((0, TendencyTerm(name="du", fn=Core.du)),
               (1, TendencyTerm(name="db", fn=Forcing.db))),
        stages=stages)
    composer.dry_run()


def test_static_advance_claim_overlap_errors_at_init(field_table):
    class A1:
        def adv(self, _state, _ctx):
            return {}

    class A2:
        def adv(self, _state, _ctx):
            return {}

    stages = (
        (0, Stage(kind=StageKind.ADVANCE, fn=A1.adv, name="adv",
                  advances=("u",))),
        (1, Stage(kind=StageKind.ADVANCE, fn=A2.adv, name="adv",
                  advances=("u",))),
    )
    with pytest.raises(AssemblyError, match="explicit order="):
        make_composer(field_table, modules=(A1(), A2()),
                      stages=stages)


# ================================================================
#  Attribution: TermEvaluationError wrapping
# ================================================================
def test_term_exception_wrapped_with_attribution(field_table):
    class Broken:
        def boom(self, _state, _ctx):
            raise ValueError("inner boom")

    terms = (
        (0, TendencyTerm(name="boom", fn=Broken.boom)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
    )
    composer = make_composer(field_table,
                             modules=(Broken(), Forcing()),
                             terms=terms)
    with pytest.raises(TermEvaluationError,
                       match="Broken/boom") as info:
        composer.dry_run()
    assert isinstance(info.value.__cause__, ValueError)
    assert "inner boom" in str(info.value.__cause__)


def test_space_mismatch_reraised_with_attribution(field_table, grid,
                                                  mx, my):
    wrong_space = mx.right * my.center

    class Staggered:
        def t(self, _state, _ctx):
            return {"u": grid.create_field(wrong_space)}

    terms = (
        (0, TendencyTerm(name="t", fn=Staggered.t)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
    )
    composer = make_composer(field_table,
                             modules=(Staggered(), Forcing()),
                             terms=terms)
    with pytest.raises(TermEvaluationError,
                       match="Staggered/t") as info:
        composer.dry_run()
    assert isinstance(info.value.__cause__, SpaceMismatchError)


def test_stage_wrong_space_write_is_attributed(field_table, grid,
                                               mx, my):
    wrong_space = mx.right * my.center

    class BadStage:
        def diagnose(self, _state, _ctx):
            return {"diag_a": grid.create_field(wrong_space)}

    stage = Stage(kind=StageKind.DIAGNOSE, fn=BadStage.diagnose,
                  name="diagnose")
    composer = make_composer(
        field_table, modules=(BadStage(), Forcing()),
        stages=((0, stage),))
    with pytest.raises(AssemblyError,
                       match=r"BadStage/diagnose.*space"):
        composer.dry_run()


# ================================================================
#  Static checks at construction
# ================================================================
def test_bound_hook_of_another_module_rejected(field_table):
    # fn bound to a DIFFERENT object than the term's slot-0 module
    core = Core()
    stranger = Forcing()
    terms = ((0, TendencyTerm(name="du", fn=stranger.db)),)
    with pytest.raises(AssemblyError, match="BOUND"):
        make_composer(field_table, modules=(core, Forcing()),
                      terms=terms)


def test_bound_hook_of_owning_module_accepted(field_table):
    # a bound method of the OWNING module normalizes to __func__ and
    # runs correctly under the (module, state, ctx) convention
    core = Core()
    forcing = Forcing()
    terms = (
        (0, TendencyTerm(name="du", fn=core.du)),
        (1, TendencyTerm(name="db", fn=forcing.db)),
    )
    composer = make_composer(field_table, modules=(core, forcing),
                             terms=terms)
    composer.dry_run()  # no aliasing error; the term evaluates


def test_bound_stage_fn_rejected(field_table):
    # stages stay strict (they have the string-fn path): a bound
    # stage fn is the aliasing trap, even of the owning module
    core = Core()
    stage = Stage(kind=StageKind.DIAGNOSE, fn=core.diagnose,
                  name="diagnose")
    with pytest.raises(AssemblyError, match="BOUND"):
        make_composer(field_table, modules=(core, Forcing()),
                      stages=((0, stage),))


def test_treatment_vs_stepper_check(field_table):
    op = FakeImplicitOp(fields=("u",))
    terms = (
        (0, TendencyTerm(name="mix", treatment=Treatment.IMPLICIT,
                         implicit=op)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
    )
    with pytest.raises(AssemblyError, match="never silently"):
        make_composer(field_table, terms=terms,
                      time_stepper=ExplicitOnlyStepper())
    # the same composition assembles under an IMEX-capable stepper
    make_composer(field_table, terms=terms,
                  time_stepper=FakeStepper())


def test_implicit_fields_must_be_prognostic(field_table):
    op = FakeImplicitOp(fields=("aux_a",))
    terms = ((0, TendencyTerm(name="mix",
                              treatment=Treatment.IMPLICIT,
                              implicit=op)),)
    with pytest.raises(AssemblyError, match="PROGNOSTIC"):
        make_composer(field_table, terms=terms)


def test_term_advances_must_be_prognostic(field_table):
    terms = ((0, TendencyTerm(name="du", fn=Core.du,
                              advances=("diag_a",))),)
    with pytest.raises(AssemblyError, match="PROGNOSTIC"):
        make_composer(field_table, terms=terms)


def test_implicit_collision_two_customs_on_one_field(field_table):
    op1 = FakeImplicitOp(fields=("u",))
    op2 = FakeImplicitOp(fields=("u", "b"))
    terms = (
        (0, TendencyTerm(name="m1", treatment=Treatment.IMPLICIT,
                         implicit=op1)),
        (1, TendencyTerm(name="m2", treatment=Treatment.IMPLICIT,
                         implicit=op2)),
    )
    with pytest.raises(ImplicitCollisionError, match="'u'"):
        make_composer(field_table, terms=terms)


def test_mergeable_family_groups_instead_of_colliding(field_table):
    op1 = FakeImplicitOp(fields=("u",), key=("family", "y"))
    op2 = FakeImplicitOp(fields=("u",), key=("family", "y"))
    terms = (
        (0, TendencyTerm(name="m1", treatment=Treatment.IMPLICIT,
                         implicit=op1)),
        (1, TendencyTerm(name="m2", treatment=Treatment.IMPLICIT,
                         implicit=op2)),
    )
    composer = make_composer(field_table, terms=terms)
    groups = composer.schedule.implicit_groups
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_bad_slot_rejected(field_table):
    terms = ((7, TendencyTerm(name="du", fn=Core.du)),)
    with pytest.raises(AssemblyError, match="slot"):
        make_composer(field_table, terms=terms)


# ================================================================
#  Advances cross-check and coverage lint
# ================================================================
def test_term_advances_cross_check(field_table):
    class Liar:
        def t(self, state, _ctx):
            return {"b": state["b"] * 1.0}

    terms = (
        (0, TendencyTerm(name="du", fn=Core.du)),
        (1, TendencyTerm(name="t", fn=Liar.t, advances=("u",))),
    )
    composer = make_composer(field_table, modules=(Core(), Liar()),
                             terms=terms)
    with pytest.raises(AssemblyError, match=r"Liar/t.*advances"):
        composer.dry_run()


def test_coverage_lint_uncovered_prognostic(field_table):
    terms = ((0, TendencyTerm(name="du", fn=Core.du)),)
    composer = make_composer(field_table, terms=terms)
    with pytest.raises(AssemblyError, match="coverage lint"):
        composer.dry_run()


def test_coverage_lint_satisfied_by_advance_claim(field_table):
    class Subcycle:
        def adv(self, state, _ctx):
            return {"b": state["b"] + 1.0}

    terms = ((0, TendencyTerm(name="du", fn=Core.du)),)
    stage = Stage(kind=StageKind.ADVANCE, fn=Subcycle.adv,
                  name="adv", advances=("b",))
    composer = make_composer(field_table,
                             modules=(Core(), Subcycle()),
                             terms=terms, stages=((1, stage),))
    composer.dry_run()


def test_coverage_lint_downgrades_under_filter(field_table):
    terms = (
        (0, TendencyTerm(name="du", fn=Core.du)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
    )
    composer = make_composer(
        field_table, terms=terms,
        term_filter=lambda key, _term: key.endswith("du"))
    with pytest.warns(UserWarning, match="coverage"):
        composer.dry_run()


def test_empty_filter_result_is_build_error(field_table):
    with pytest.raises(AssemblyError, match="filter"):
        make_composer(field_table,
                      term_filter=lambda _key, _term: False)


# ================================================================
#  Abstract evaluation: dry_run fires no eager compiles
# ================================================================
def test_dry_run_does_not_eagerly_compile(field_table,
                                          compile_counter):
    # the whole hook pass runs under a single jax.eval_shape trace, so
    # no term/stage hook dispatches its own one-shot eager compile;
    # before the fix each traced eager op counted (>100 for a real
    # model), now the abstract-evaluation trace itself is the only
    # event. Build the composer first, then reset immediately before
    # the measured dry_run (eager ops also trace on first occurrence).
    composer = make_composer(field_table)
    compile_counter.reset()
    composer.dry_run()
    assert compile_counter.count <= 2


# ================================================================
#  Kind ordering
# ================================================================
def test_schedule_kind_order_self_update_first(field_table):
    class Full:
        def up(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

        def diag(self, state, _ctx):
            return {"diag_a": state["aux_a"] * 1.0}

        def adv(self, state, _ctx):
            return {"aux_a": state["aux_a"] + 1.0}

        def proj(self, state, _ctx):
            return {"u": state["u"] * 0.5}

        def acc(self, state, _ctx):
            return {"diag_a": state["diag_a"] + 1.0}

    stages = (
        (0, Stage(kind=StageKind.DIAGNOSTIC, fn=Full.acc,
                  name="acc")),
        (0, Stage(kind=StageKind.CONSTRAINT, fn=Full.proj,
                  name="proj")),
        (0, Stage(kind=StageKind.ADVANCE, fn=Full.adv, name="adv")),
        (0, Stage(kind=StageKind.DIAGNOSE, fn=Full.diag,
                  name="diag")),
        (0, Stage(kind=StageKind.SELF_UPDATE, fn=Full.up,
                  name="up")),
    )
    terms = (
        (0, TendencyTerm(name="du", fn=Core.du)),
        (1, TendencyTerm(name="db", fn=Forcing.db)),
    )
    composer = make_composer(field_table,
                             modules=(Full(), Forcing()),
                             terms=terms, stages=stages)
    kinds = [("TERM" if e.is_term else e.kind.name)
             for e in composer.schedule.entries]
    assert kinds == ["SELF_UPDATE", "DIAGNOSE", "TERM", "TERM",
                     "ADVANCE", "CONSTRAINT", "DIAGNOSTIC"]
    assert kinds[0] == "SELF_UPDATE"  # load-bearing: S1 first


# ================================================================
#  Composed body: purity and jit
# ================================================================
def test_composed_body_is_pure_and_deterministic(field_table, grid,
                                                 cc):
    composer = make_composer(field_table)
    body = composer.compose()
    modules = (Core(), Forcing())
    state = _full_state(field_table).replace(
        u=grid.create_field(cc, data=jnp.linspace(
            0.0, 1.0, 32).reshape(8, 4)),
        b=grid.create_field(cc, data=jnp.linspace(
            1.0, 2.0, 32).reshape(8, 4)),
    )
    ctx = make_ctx()
    out1, sums1 = body(state, modules, ctx)
    out2, sums2 = body(state, modules, ctx)
    for name in out1.component_names:
        assert jnp.array_equal(out1[name].data, out2[name].data)
    assert jnp.array_equal(sums1.explicit["u"].data,
                           sums2.explicit["u"].data)
    assert jnp.array_equal(sums1.explicit["b"].data,
                           sums2.explicit["b"].data)
    # the sums are the accumulated term contributions
    assert jnp.allclose(sums1.explicit["u"].data,
                        state["b"].data * 0.5)
    assert jnp.allclose(sums1.explicit["b"].data,
                        state["u"].data * 0.25 + 1.0)


def test_composed_body_works_under_jit(field_table, grid, cc):
    composer = make_composer(field_table)
    body = composer.compose()
    modules = (Core(), Forcing())
    state = _full_state(field_table).replace(
        b=grid.create_field(cc, data=jnp.full((8, 4), 2.0)))
    ctx = make_ctx()
    eager_state, eager_sums = body(state, modules, ctx)

    jitted = jax.jit(lambda s, c: body(s, modules, c))
    jit_state, jit_sums = jitted(state, ctx)
    assert (jax.tree_util.tree_structure(jit_state)
            == jax.tree_util.tree_structure(eager_state))
    assert jnp.allclose(jit_sums.explicit["u"].data,
                        eager_sums.explicit["u"].data)
    assert jnp.allclose(jit_state["u"].data, eager_state["u"].data)


# ================================================================
#  Templates and stubs
# ================================================================
def test_tendency_template_zero_prognostic_vector(field_table):
    composer = make_composer(field_table)
    template = composer.tendency_template()
    assert template.component_names == ("u", "b")
    assert jnp.allclose(template["u"].data, 0.0)
    assert jnp.allclose(template["b"].data, 0.0)


def test_tendency_fn_is_wave7_stub(field_table):
    composer = make_composer(field_table)
    with pytest.raises(NotImplementedError, match=r"2\.8"):
        composer.tendency_fn()


def test_stage_method_name_resolution(field_table):
    # Stage.fn may be a method-name string, resolved at collection
    stage = Stage(kind=StageKind.DIAGNOSE, fn="diagnose")
    composer = make_composer(field_table, stages=((0, stage),))
    composer.dry_run()
    entry = composer.schedule.kind_entries(StageKind.DIAGNOSE)[0]
    assert entry.key == "Core/diagnose"
    assert entry.fn is Core.diagnose


def test_stage_unknown_method_name_errors(field_table):
    stage = Stage(kind=StageKind.DIAGNOSE, fn="nope")
    with pytest.raises(AssemblyError, match="nope"):
        make_composer(field_table, stages=((0, stage),))


def test_self_update_unknown_reads_errors(field_table):
    stage = Stage(kind=StageKind.SELF_UPDATE, fn=Core.update_aux,
                  name="update_aux", reads=("ghost",))
    with pytest.raises(AssemblyError, match="ghost"):
        make_composer(field_table, stages=((0, stage),))
