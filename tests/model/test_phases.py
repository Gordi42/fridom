"""Tests for the phase axis vocabulary (model/phases.py).

Covers the three ``Phases`` spellings and their resolution against a
PROGNOSTIC name tuple (the role-derived staggered split, the explicit
groups, the one-group total), every validation refusal, the
``PhaseView`` surface, and the ``fields_in_phase`` hook helper —
including its load-bearing identity return on the unphased path.
"""
import pytest

from fridom.model.context import StepContext
from fridom.model.errors import AssemblyError
from fridom.model.phases import Phases, PhaseView, fields_in_phase

PROG = ("u", "v", "ps", "b")


# ================================================================
#  Construction
# ================================================================
def test_explicit_groups_round_trip_in_repr():
    phases = Phases(("u", "v"), ("b",))
    assert phases.groups == (("u", "v"), ("b",))
    assert phases.rule is None
    assert repr(phases) == "Phases(('u', 'v'), ('b',))"


def test_derived_spellings_carry_a_rule_and_no_groups():
    assert Phases.staggered().rule == "staggered"
    assert Phases.total().rule == "total"
    assert Phases.staggered().groups is None
    assert repr(Phases.total()) == "Phases.total()"


def test_no_group_is_refused_with_the_total_remedy():
    with pytest.raises(ValueError, match=r"Phases\.total"):
        Phases()


def test_a_bare_string_group_is_refused():
    # Phases("u", "v") would silently make two one-letter groups
    with pytest.raises(TypeError, match="bare string"):
        Phases("uv", ("b",))


def test_a_non_string_name_is_refused():
    with pytest.raises(TypeError, match="field-name"):
        Phases((1, 2), ("b",))


def test_value_semantics_hash_and_compare_by_structure():
    assert Phases(("u",), ("b",)) == Phases(("u",), ("b",))
    assert hash(Phases(("u",), ("b",))) == hash(
        Phases(("u",), ("b",)))
    assert Phases.staggered() == Phases.staggered()
    assert Phases.staggered() != Phases.total()
    assert Phases(("u",), ("b",)) != Phases.staggered()
    assert Phases.total() != "total"


# ================================================================
#  Resolution
# ================================================================
def test_total_resolves_to_one_group_holding_everything():
    groups = Phases.total().resolve(PROG)
    assert groups == (frozenset(PROG),)


def test_staggered_takes_velocity_plus_claims_as_group_zero():
    # the role-derived rule: ps joins group 0 through the ADVANCE /
    # CONSTRAINT claim, without ever being named
    groups = Phases.staggered().resolve(
        PROG, velocity=("u", "v"), claimed=("ps",))
    assert groups == (frozenset({"u", "v", "ps"}), frozenset({"b"}))


def test_staggered_ignores_claims_on_non_prognostic_names():
    groups = Phases.staggered().resolve(
        ("u", "b"), velocity=("u",), claimed=("w", "eta"))
    assert groups == (frozenset({"u"}), frozenset({"b"}))


def test_staggered_without_a_tracer_group_is_refused():
    with pytest.raises(AssemblyError, match="single non-empty group"):
        Phases.staggered().resolve(
            ("u", "v"), velocity=("u", "v"))


def test_staggered_without_a_momentum_group_is_refused():
    with pytest.raises(AssemblyError, match="single non-empty group"):
        Phases.staggered().resolve(("b", "c"))


def test_explicit_groups_resolve_to_frozensets_in_order():
    groups = Phases(("ps", "u", "v"), ("b",)).resolve(PROG)
    assert groups == (frozenset({"u", "v", "ps"}), frozenset({"b"}))


def test_explicit_group_naming_an_unknown_field_is_refused():
    with pytest.raises(AssemblyError, match="not declared PROGNOSTIC"):
        Phases(("u", "eta"), ("b",)).resolve(PROG)


def test_explicit_group_repeating_a_name_is_refused():
    with pytest.raises(AssemblyError, match="repeats a field name"):
        Phases(("u", "u"), ("v", "ps", "b")).resolve(PROG)


def test_a_field_in_two_groups_is_refused():
    with pytest.raises(AssemblyError, match="more than one phase"):
        Phases(("u", "v", "ps"), ("u", "b")).resolve(PROG)


def test_a_field_in_no_group_is_refused():
    with pytest.raises(AssemblyError, match="belong to no phase"):
        Phases(("u", "v"), ("b",)).resolve(PROG)


def test_an_empty_group_is_refused():
    with pytest.raises(AssemblyError, match="are empty"):
        Phases(("u", "v", "ps", "b"), ()).resolve(PROG)


def test_a_single_explicit_group_is_the_unphased_spelling():
    # one group is legal (it IS the unphased path) and needs no
    # non-emptiness check beyond the exact-partition rule
    groups = Phases(PROG).resolve(PROG)
    assert len(groups) == 1


def test_total_resolves_on_a_field_free_composition():
    assert Phases.total().resolve(()) == (frozenset(),)


# ================================================================
#  PhaseView
# ================================================================
def test_phase_view_exposes_index_and_fields():
    view = PhaseView(1, ["b", "c"])
    assert view.index == 1
    assert view.fields == frozenset({"b", "c"})
    assert "PhaseView(1" in repr(view)


def test_phase_view_restrict_keeps_the_callers_order():
    view = PhaseView(0, {"u", "b"})
    assert view.restrict(("b", "v", "u")) == ("b", "u")


def test_phase_view_is_value_hashable_for_the_treedef_aux():
    assert PhaseView(0, ("u",)) == PhaseView(0, ["u"])
    assert hash(PhaseView(0, ("u",))) == hash(PhaseView(0, ["u"]))
    assert PhaseView(0, ("u",)) != PhaseView(1, ("u",))
    assert PhaseView(0, ("u",)) != "phase"


# ================================================================
#  fields_in_phase (the hook helper)
# ================================================================
def _ctx(phase):
    return StepContext(params={}, clock=0.0, dt=1.0, stage_dt=1.0,
                       phase=phase)


def test_fields_in_phase_returns_the_declared_tuple_unphased():
    # IDENTITY, not equality: the unphased iteration must be the
    # literal declared tuple (the bitwise guarantee)
    names = ("u", "v", "b")
    assert fields_in_phase(names, _ctx(None)) is names


def test_fields_in_phase_masks_to_the_phase():
    names = ("u", "v", "b")
    view = PhaseView(0, {"u", "v"})
    assert fields_in_phase(names, _ctx(view)) == ("u", "v")


def test_fields_in_phase_can_be_empty():
    view = PhaseView(1, {"b"})
    assert fields_in_phase(("u", "v"), _ctx(view)) == ()


def test_fields_in_phase_reads_a_duck_typed_context_as_unphased():
    # hooks are called directly with a bare ctx (params only) in
    # tests and notebooks; "no phase attribute" means "no axis"
    class BareCtx:
        params = {}  # noqa: RUF012

    names = ("u", "b")
    assert fields_in_phase(names, BareCtx) is names
