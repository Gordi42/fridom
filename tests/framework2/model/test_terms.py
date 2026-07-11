"""Tests for the term vocabulary (framework2/model/terms.py)."""
import dataclasses

import pytest

from fridom.framework2.model.terms import (
    EXPLICIT,
    IMPLICIT,
    TERM_ATTRIBUTE,
    TendencyTerm,
    Treatment,
    term,
)


class DummyOperator:

    """A minimal ImplicitOperator-satisfying stand-in."""

    def __init__(self, fields=("w",)):
        self.fields = tuple(fields)

    def apply(self, _module, _state, _ctx):
        return {}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        raise NotImplementedError


# ================================================================
#  Treatment
# ================================================================
def test_treatment_members():
    assert list(Treatment) == [Treatment.EXPLICIT, Treatment.IMPLICIT]


def test_module_aliases_are_the_members():
    assert EXPLICIT is Treatment.EXPLICIT
    assert IMPLICIT is Treatment.IMPLICIT


# ================================================================
#  TendencyTerm construction
# ================================================================
def test_records_attributes():
    def fn(_module, _state, _ctx):
        return {}

    declared = TendencyTerm(
        name="momentum", fn=fn, treatment=Treatment.EXPLICIT,
        advances=("u", "v"), transports=("b",), linear=True)
    assert declared.name == "momentum"
    assert declared.fn is fn
    assert declared.treatment is Treatment.EXPLICIT
    assert declared.advances == ("u", "v")
    assert declared.transports == ("b",)
    assert declared.implicit is None
    assert declared.linear is True


def test_defaults():
    declared = TendencyTerm(name="t", fn=lambda _m, _s, _c: {})
    assert declared.treatment is Treatment.EXPLICIT
    assert declared.advances is None
    assert declared.transports == ()
    assert declared.implicit is None
    assert declared.linear is False


def test_is_frozen():
    declared = TendencyTerm(name="t", fn=lambda _m, _s, _c: {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        declared.name = "other"


def test_normalizes_name_lists_to_tuples():
    declared = TendencyTerm(
        name="t", fn=lambda _m, _s, _c: {},
        advances=["u", "v"], transports=["b"])
    assert declared.advances == ("u", "v")
    assert declared.transports == ("b",)


def test_advances_none_stays_none():
    declared = TendencyTerm(name="t", fn=lambda _m, _s, _c: {})
    assert declared.advances is None


def test_implicit_term_may_omit_fn():
    operator = DummyOperator()
    declared = TendencyTerm(
        name="vertical", treatment=Treatment.IMPLICIT,
        implicit=operator)
    assert declared.fn is None
    assert declared.implicit is operator


def test_rejects_term_without_behavior():
    with pytest.raises(ValueError, match="declares no behavior"):
        TendencyTerm(name="empty")


def test_rejects_non_treatment():
    with pytest.raises(TypeError, match="Treatment member"):
        TendencyTerm(name="t", fn=lambda _m, _s, _c: {},
                     treatment="implicit")


def test_rejects_empty_name():
    with pytest.raises(TypeError, match="non-empty string"):
        TendencyTerm(name="", fn=lambda _m, _s, _c: {})


def test_rejects_non_callable_fn():
    with pytest.raises(TypeError, match="must be callable"):
        TendencyTerm(name="t", fn=42)


def test_rejects_non_protocol_implicit():
    with pytest.raises(TypeError, match="ImplicitOperator protocol"):
        TendencyTerm(name="t", implicit=lambda: None)


def test_repr_is_compact():
    declared = TendencyTerm(
        name="buoyancy", fn=lambda _m, _s, _c: {}, advances=("w",),
        linear=True)
    text = repr(declared)
    assert "buoyancy" in text
    assert "EXPLICIT" in text
    assert "advances=('w',)" in text
    assert "linear=True" in text


# ================================================================
#  The @fr.term decorator
# ================================================================
class Stratification:

    """A module-like host with stamped term methods."""

    @term
    def restoring(self, _state, _ctx):
        return {"b": "increment"}

    @term(name="force", advances=("w",), transports=("b",),
          linear=True)
    def buoyancy_force(self, _state, _ctx):
        return {"w": "increment"}


def test_bare_form_stamps_a_term():
    declared = getattr(Stratification.restoring, TERM_ATTRIBUTE)
    assert isinstance(declared, TendencyTerm)
    assert declared.name == "restoring"
    assert declared.treatment is Treatment.EXPLICIT
    assert declared.linear is False


def test_parenthesized_form_records_arguments():
    declared = getattr(Stratification.buoyancy_force, TERM_ATTRIBUTE)
    assert declared.name == "force"
    assert declared.advances == ("w",)
    assert declared.transports == ("b",)
    assert declared.linear is True


def test_fn_is_stored_unbound():
    declared = getattr(Stratification.restoring, TERM_ATTRIBUTE)
    # the stamped fn is the plain class-body function, not a bound
    # method (the aliasing rule); binding happens per call, from
    # the module slot, inside the composer
    assert declared.fn is Stratification.__dict__["restoring"]
    assert not hasattr(declared.fn, "__self__")


def test_decorated_method_stays_plainly_callable():
    module = Stratification()
    assert module.restoring("state", "ctx") == {"b": "increment"}
    assert module.buoyancy_force("state", "ctx") == {"w": "increment"}


def test_stamps_preserve_definition_order():
    stamped = [name for name, attr in vars(Stratification).items()
               if hasattr(attr, TERM_ATTRIBUTE)]
    assert stamped == ["restoring", "buoyancy_force"]


def test_decorator_rejects_bad_records_at_decoration_time():
    with pytest.raises(TypeError, match="Treatment member"):
        @term(treatment="explicit")
        def bad(_self, _state, _ctx):
            return {}
