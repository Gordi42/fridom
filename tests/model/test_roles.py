"""Tests for the role markers (model/roles.py)."""
import pytest

from fridom.model.roles import ADVECTED, TRACER, Role, Velocity


# ================================================================
#  Role
# ================================================================
def test_equality_and_hash_by_key():
    assert Role("mybgc.nutrient") == Role("mybgc.nutrient")
    assert hash(Role("mybgc.nutrient")) == hash(
        Role("mybgc.nutrient"))
    assert Role("mybgc.nutrient") != Role("mybgc.oxygen")
    assert Role("a.b") != "a.b"


def test_roles_are_selection_keys():
    selections = {Role("mybgc.nutrient"): ("no3", "po4")}
    assert selections[Role("mybgc.nutrient")] == ("no3", "po4")
    assert len({Role("a.b"), Role("a.b"), Role("a.c")}) == 2


def test_key_property():
    assert Role("mybgc.nutrient").key == "mybgc.nutrient"


def test_keys_are_namespaced():
    with pytest.raises(ValueError, match="namespaced"):
        Role("advected")
    with pytest.raises(TypeError, match="non-empty"):
        Role("")
    with pytest.raises(TypeError, match="non-empty"):
        Role(42)


def test_repr_round_trips():
    namespace = {"Role": Role, "Velocity": Velocity}
    for role in (Role("mybgc.nutrient"), Velocity("x"), ADVECTED):
        assert eval(repr(role), namespace) == role  # noqa: S307


# ================================================================
#  The framework constants
# ================================================================
def test_framework_role_keys():
    assert Role("fridom.advected") == ADVECTED
    assert Role("fridom.tracer") == TRACER
    assert ADVECTED != TRACER


# ================================================================
#  Velocity
# ================================================================
def test_velocity_component_and_key():
    u = Velocity("x")
    assert u.component == "x"
    assert u.key == ("fridom.velocity", "x")


def test_velocity_hashes_by_pair():
    assert Velocity("x") == Velocity("x")
    assert hash(Velocity("x")) == hash(Velocity("x"))
    assert Velocity("x") != Velocity("y")
    assert len({Velocity("x"), Velocity("x"), Velocity("y")}) == 2


def test_velocity_never_equals_a_flat_role():
    # the pair key can never collide with a flat string key
    assert Velocity("x") != ADVECTED
    assert isinstance(Velocity("x"), Role)


def test_velocity_component_validation():
    with pytest.raises(TypeError, match="non-empty"):
        Velocity("")
    with pytest.raises(TypeError, match="non-empty"):
        Velocity(0)


def test_velocity_is_final():
    with pytest.raises(TypeError, match="final"):
        class EdgeNormal(Velocity):
            pass
