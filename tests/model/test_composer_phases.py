"""Phase resolution in the composer (model/composer.py).

Prefix-mirrored shard of ``test_composer.py``: the role-derived
partition off the field table, the per-entry membership rules, and
every taught refusal of the phase axis (unsupported stepper,
straddling term / claim / implicit merge group, an out-of-range pin,
the phase-aware coverage and overlap lints). Self-contained per the
AGENTS oversized-module rule.
"""
from typing import NamedTuple

import pytest

from fridom.model.composer import TendencyComposer
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.phases import Phases
from fridom.model.roles import Velocity
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
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

    def straddle(self, state, _ctx):
        return {"u": state["u"] * 0.5, "b": state["b"] * 0.5}

    def solve(self, state, _ctx):
        return {"ps": state["ps"] + 1.0}

    def claim_both(self, state, _ctx):
        return {"u": state["u"], "b": state["b"]}

    def project(self, state, _ctx):
        return {"u": state["u"] * 2.0}

    def touch_aux(self, state, _ctx):
        return {"aux_a": state["aux_a"] + 1.0}

    def touch_aux_again(self, state, _ctx):
        return {"aux_a": state["aux_a"] + 2.0}


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


class SeparableImplicitOp(FakeImplicitOp):

    """A per-field-independent operator (the VerticalDiffusion seam)."""

    def restricted_to(self, fields):
        keep = frozenset(fields)
        return SeparableImplicitOp(
            tuple(n for n in self.fields if n in keep))


class PhasedStepper:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})
    supports_phases = True


class UnphasedStepper:
    supported_treatments = frozenset({Treatment.EXPLICIT})
    supports_phases = False


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
        Record("u", cc, Lifecycle.PROGNOSTIC, 0, (Velocity("x"),)),
        Record("ps", cc, Lifecycle.PROGNOSTIC, 0),
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
        Record("aux_a", cc, Lifecycle.AUXILIARY, 0),
    ))


DEFAULT_TERMS = (
    (0, TendencyTerm(name="du", fn=Core.du)),
    (1, TendencyTerm(name="db", fn=Tracer.db)),
)

#: the explicit momentum/tracer split (used where a claim on b would
#: otherwise pull b into the role-derived group 0)
SPLIT = Phases(("u", "ps"), ("b",))

# the barotropic claim: what puts ps in group 0 under staggered()
SOLVE_STAGE = (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.solve,
                        name="solve", advances=("ps",)))


def compose(field_table, *, terms=DEFAULT_TERMS,
            stages=(SOLVE_STAGE,), phases=None, stepper=None,
            term_filter=None, allow_unadvanced=()):
    composer = TendencyComposer(
        field_table=field_table, modules=(Core(), Tracer()),
        terms=terms, stages=stages,
        time_stepper=PhasedStepper() if stepper is None else stepper,
        binding_table=None, phases=phases, term_filter=term_filter,
        allow_unadvanced=allow_unadvanced)
    composer.dry_run()
    return composer.schedule


def membership(schedule, key):
    return next(e.active_phases for e in schedule.entries
                if e.key == key)


# ================================================================
#  Resolution
# ================================================================
def test_staggered_derives_momentum_plus_the_claim(field_table):
    schedule = compose(field_table, phases=Phases.staggered())
    assert schedule.phases == (frozenset({"u", "ps"}),
                               frozenset({"b"}))


def test_terms_land_in_the_phase_owning_their_writes(field_table):
    schedule = compose(field_table, phases=Phases.staggered())
    assert membership(schedule, "Core/du") == (0,)
    assert membership(schedule, "Tracer/db") == (1,)


def test_a_claiming_constraint_lands_in_its_claims_phase(
        field_table):
    schedule = compose(field_table, phases=Phases.staggered())
    assert membership(schedule, "Core/solve") == (0,)


def test_self_update_and_diagnose_run_in_every_phase(field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.touch_aux,
                        name="touch_aux")))
    schedule = compose(field_table, stages=stages,
                       phases=Phases.staggered())
    assert membership(schedule, "Core/touch_aux") is None


def test_an_unclaimed_constraint_follows_its_write_set(field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.project,
                        name="project", order=1)))
    schedule = compose(field_table, stages=stages,
                       phases=Phases.staggered())
    # project writes u only -> the momentum phase only
    assert membership(schedule, "Core/project") == (0,)


def test_a_per_phase_term_runs_in_every_phase_it_touches(
        field_table):
    terms = (*DEFAULT_TERMS,
             (0, TendencyTerm(name="joint", fn=Core.straddle,
                              per_phase=True)))
    schedule = compose(field_table, terms=terms,
                       phases=Phases.staggered())
    assert membership(schedule, "Core/joint") == (0, 1)
    assert next(e.per_phase for e in schedule.entries
                if e.key == "Core/joint")


def test_an_explicit_pin_overrides_the_kind_rule(field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.touch_aux,
                        name="touch_aux", phase=1)))
    schedule = compose(field_table, stages=stages,
                       phases=Phases.staggered())
    assert membership(schedule, "Core/touch_aux") == (1,)


def test_total_installs_nothing_and_keeps_the_plain_entries(
        field_table):
    schedule = compose(field_table, phases=Phases.total())
    assert not schedule.phased
    assert all(entry.active_phases is None
               for entry in schedule.entries)


def test_a_pin_is_inert_on_the_unphased_path(field_table):
    # a module may declare Stage(phase=1) and still assemble without
    # an axis (the split-explicit snapshot's phase=0 precedent)
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.touch_aux,
                        name="touch_aux", phase=3)))
    schedule = compose(field_table, stages=stages)
    assert not schedule.phased


# ================================================================
#  Refusals
# ================================================================
def test_an_axis_under_an_unphased_stepper_is_refused(field_table):
    with pytest.raises(AssemblyError, match="supports_phases=False"):
        compose(field_table, phases=Phases.staggered(),
                stepper=UnphasedStepper())


def test_a_stepper_without_the_attribute_is_refused(field_table):
    class Bare:
        supported_treatments = frozenset({Treatment.EXPLICIT})

    with pytest.raises(AssemblyError, match="does not support"):
        compose(field_table, phases=Phases.staggered(), stepper=Bare())


def test_a_straddling_term_without_per_phase_is_refused(field_table):
    terms = (*DEFAULT_TERMS,
             (0, TendencyTerm(name="joint", fn=Core.straddle)))
    with pytest.raises(AssemblyError, match="per_phase=True"):
        compose(field_table, terms=terms, phases=Phases.staggered())


def test_a_straddling_claim_is_refused_with_the_pin_remedy(
        field_table):
    # explicit groups, not staggered(): a claim on b would otherwise
    # PULL b into group 0 (the role-derived rule reads the claims)
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.claim_both,
                        name="claim_both", order=1,
                        advances=("u", "b"))))
    with pytest.raises(AssemblyError, match="different phase groups"):
        compose(field_table, stages=stages, phases=SPLIT)


def test_a_straddling_implicit_merge_group_is_refused(field_table):
    terms = (*DEFAULT_TERMS,
             (0, TendencyTerm(name="mix",
                              treatment=Treatment.IMPLICIT,
                              implicit=FakeImplicitOp(("u", "b")))))
    with pytest.raises(AssemblyError, match="atomic under by-variable"):
        compose(field_table, terms=terms, phases=Phases.staggered())


def test_an_out_of_range_pin_is_refused(field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.touch_aux,
                        name="touch_aux", phase=4)))
    with pytest.raises(AssemblyError, match="out of range"):
        compose(field_table, stages=stages, phases=Phases.staggered())


def test_an_empty_group_is_refused_through_the_composer(field_table):
    with pytest.raises(AssemblyError, match="are empty"):
        compose(field_table, phases=Phases(("u", "ps", "b"), ()))


def test_a_field_advanced_only_in_another_phase_is_refused(
        field_table):
    # the per-phase coverage lint: the claim on b is pinned to the
    # momentum phase, so nothing advances b in its own phase
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.claim_both,
                        name="claim_both", order=1,
                        advances=("b",), phase=0)))
    terms = ((0, TendencyTerm(name="du", fn=Core.du)),)
    with pytest.raises(AssemblyError, match="nothing advances them"):
        compose(field_table, terms=terms, stages=stages,
                phases=SPLIT)


def test_the_phase_coverage_lint_honours_allow_unadvanced(
        field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.claim_both,
                        name="claim_both", order=1,
                        advances=("b",), phase=0)))
    terms = ((0, TendencyTerm(name="du", fn=Core.du)),)
    schedule = compose(field_table, terms=terms, stages=stages,
                       phases=SPLIT, allow_unadvanced=("b",))
    assert schedule.phased


def test_a_term_filter_downgrades_the_phase_lint_to_a_warning(
        field_table):
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.claim_both,
                        name="claim_both", order=1,
                        advances=("b",), phase=0)))
    with pytest.warns(UserWarning, match="nothing advances them"):
        compose(field_table, stages=stages, phases=SPLIT,
                term_filter=lambda key, _term: key != "Tracer/db")


# ================================================================
#  The phase-aware overlap lint
# ================================================================
def test_same_kind_stages_in_different_phases_no_longer_collide(
        field_table):
    # two SELF_UPDATE stages rewriting the SAME aux field at the same
    # order: an assembly error unphased, legal once each is pinned to
    # its own phase (the realized-geometry pattern)
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.SELF_UPDATE, fn=Core.touch_aux,
                        name="touch_aux", phase=0)),
              (0, Stage(kind=StageKind.SELF_UPDATE,
                        fn=Core.touch_aux_again,
                        name="touch_aux_again", phase=1)))
    schedule = compose(field_table, stages=stages,
                       phases=Phases.staggered())
    assert membership(schedule, "Core/touch_aux") == (0,)
    assert membership(schedule, "Core/touch_aux_again") == (1,)
    with pytest.raises(AssemblyError, match="overlapping write"):
        compose(field_table, stages=stages)


def test_a_claim_on_a_tracer_pulls_it_into_the_momentum_group(
        field_table):
    # the role-derived rule reads the CLAIMS as well as the roles, so
    # a tracer advanced by a stage joins group 0 — here that empties
    # the tracer group and staggered() refuses (name the groups
    # explicitly if the claim is deliberate)
    stages = (SOLVE_STAGE,
              (0, Stage(kind=StageKind.CONSTRAINT, fn=Core.claim_both,
                        name="claim_both", order=1,
                        advances=("u", "b"))))
    with pytest.raises(AssemblyError, match="single non-empty group"):
        compose(field_table, stages=stages, phases=Phases.staggered())


def test_a_separable_straddling_group_is_split_per_phase(
        field_table):
    # VerticalDiffusion's seam: apply/solve loop per field over
    # independent bands, so the merge group is SPLIT, not refused
    terms = (*DEFAULT_TERMS,
             (0, TendencyTerm(name="mix",
                              treatment=Treatment.IMPLICIT,
                              implicit=SeparableImplicitOp(
                                  ("u", "b")))))
    schedule = compose(field_table, terms=terms,
                       phases=Phases.staggered())
    assert schedule.implicit_phases == (0, 1)
    assert [tuple(op.fields) for op, _slot
            in schedule.implicit_merged] == [("u",), ("b",)]
    assert len(schedule.implicit_groups) == 2


def test_the_split_is_skipped_when_the_group_lives_in_one_phase(
        field_table):
    terms = (*DEFAULT_TERMS,
             (0, TendencyTerm(name="mix",
                              treatment=Treatment.IMPLICIT,
                              implicit=SeparableImplicitOp(("b",)))))
    schedule = compose(field_table, terms=terms,
                       phases=Phases.staggered())
    assert schedule.implicit_phases == (1,)
    assert [tuple(op.fields) for op, _slot
            in schedule.implicit_merged] == [("b",)]
