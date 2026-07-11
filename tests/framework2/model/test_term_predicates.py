"""Tests for fr.terms predicates (model/term_predicates.py)."""
import pytest

from fridom.framework2.model import term_predicates as terms
from fridom.framework2.model.term_predicates import linearize
from fridom.framework2.model.terms import TendencyTerm, Treatment


def _fn(_module, _state, _ctx):
    return {}


def _term(name, *, linear=False, treatment=Treatment.EXPLICIT,
          advances=None, implicit=None):
    return TendencyTerm(name=name, fn=_fn, treatment=treatment,
                        advances=advances, linear=linear,
                        implicit=implicit)


class ModuleA:
    pass


class ModuleB(ModuleA):
    pass


# ================================================================
#  Leaves
# ================================================================
def test_linear_leaf():
    assert terms.linear("A/t", _term("t", linear=True))
    assert not terms.linear("A/t", _term("t", linear=False))


def test_explicit_and_implicit():
    ex = _term("t", treatment=Treatment.EXPLICIT)
    assert terms.explicit("A/t", ex)
    assert not terms.implicit("A/t", ex)


def test_owned_by_isinstance_with_module():
    pred = terms.owned_by(ModuleA)
    assert pred("ModuleB/t", _term("t"), ModuleB())  # subclass matches
    assert not pred("Other/t", _term("t"), object())


def test_owned_by_qualname_without_module():
    pred = terms.owned_by(ModuleA)
    assert pred("ModuleA/t", _term("t"))  # falls back to name match
    assert not pred("Other/t", _term("t"))


def test_owned_by_rejects_non_type():
    with pytest.raises(TypeError, match="module type"):
        terms.owned_by(ModuleA())


def test_named():
    pred = terms.named("A/keep", "B/keep")
    assert pred("A/keep", _term("keep"))
    assert not pred("A/drop", _term("drop"))
    assert pred.referenced_names() == frozenset({"A/keep", "B/keep"})


def test_named_requires_a_key():
    with pytest.raises(ValueError, match="at least one"):
        terms.named()


def test_advancing():
    pred = terms.advancing("u", "v")
    assert pred("A/t", _term("t", advances=("u",)))
    assert not pred("A/t", _term("t", advances=("w",)))
    assert not pred("A/t", _term("t", advances=None))


def test_advancing_requires_a_field():
    with pytest.raises(ValueError, match="at least one"):
        terms.advancing()


# ================================================================
#  Combinators + fingerprint tokens
# ================================================================
def test_and_or_not():
    lin = terms.linear
    imp = terms.implicit
    t_lin = _term("t", linear=True, treatment=Treatment.EXPLICIT)
    assert (lin & ~imp)("A/t", t_lin)
    assert (lin | imp)("A/t", t_lin)
    assert not (lin & imp)("A/t", t_lin)


def test_fingerprint_tokens_are_stable():
    assert terms.linear.token == "linear"  # noqa: S105
    assert terms.owned_by(ModuleA).fingerprint_token().endswith(
        "ModuleA)")
    combo = terms.linear & ~terms.owned_by(ModuleA)
    assert combo.fingerprint_token() == (
        "(linear & ~owned_by("
        f"{ModuleA.__qualname__}))")


def test_all_leaf_tokens():
    assert terms.explicit.fingerprint_token() == "explicit"
    assert terms.implicit.fingerprint_token() == "implicit"
    assert terms.named("B/y", "A/x").fingerprint_token() == (
        "named(A/x,B/y)")  # sorted
    assert terms.advancing("v", "u").fingerprint_token() == (
        "advancing(u,v)")  # sorted
    assert (terms.linear | terms.implicit).fingerprint_token() == (
        "(linear | implicit)")


def test_referenced_names_traverses_tree():
    combo = (terms.named("A/x") | terms.linear) & ~terms.named("B/y")
    assert combo.referenced_names() == frozenset({"A/x", "B/y"})


def test_linearize_explicit_name():
    captured = {}

    class FakeModel:
        def variant(self, *, term_filter, name):
            captured["filter"] = term_filter
            captured["name"] = name
            return "variant"

    result = linearize(FakeModel(), name="custom/linear")
    assert result == "variant"
    assert captured["name"] == "custom/linear"
    assert captured["filter"] is terms.linear


def test_repr():
    assert repr(terms.linear) == "fr.terms.linear"


def test_combinator_notimplemented_on_bad_operand():
    assert terms.linear.__and__(5) is NotImplemented
    assert terms.linear.__or__("x") is NotImplemented
