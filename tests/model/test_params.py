"""Tests for fridom.model.params (ParamName + registry)."""
import pytest

from fridom.model import params
from fridom.model.params import ParamName


# ================================================================
#  ParamName: str subclass semantics
# ================================================================
def test_paramname_is_a_str():
    name = ParamName("coriolis.f0", units="1/s")
    assert isinstance(name, str)
    assert name == "coriolis.f0"
    assert hash(name) == hash("coriolis.f0")


def test_paramname_and_plain_string_hit_the_same_mapping_key():
    # the design's two spellings — indexing by the plain dotted
    # string and by the registry constant — hit the same key
    table = {params.CORIOLIS_F0: 1e-4}
    assert table["coriolis.f0"] == 1e-4

    table = {"coriolis.f0": 2e-4}
    assert table[params.CORIOLIS_F0] == 2e-4


def test_paramname_equality_ignores_registry_documentation():
    bare = ParamName("coriolis.f0")
    documented = ParamName("coriolis.f0", units="1/s", hint="x",
                           no_default=True)
    assert bare == documented


def test_paramname_str_methods_still_work():
    assert params.TIME_STEP.split(".") == ["stepper", "dt"]
    assert params.CORIOLIS_F0.startswith("coriolis.")


# ================================================================
#  ParamName: registry documentation attributes
# ================================================================
def test_paramname_attributes():
    name = ParamName("mypkg.alpha", units="m/s", hint="see MyModule",
                     no_default=True)
    assert name.units == "m/s"
    assert name.hint == "see MyModule"
    assert name.no_default is True


def test_paramname_attribute_defaults():
    name = ParamName("mypkg.alpha")
    assert name.units == "n/a"
    assert name.hint == ""
    assert name.no_default is False


def test_paramname_attributes_are_read_only():
    name = ParamName("mypkg.alpha")
    with pytest.raises(AttributeError):
        name.units = "m"
    with pytest.raises(AttributeError):
        name.no_default = True


def test_dotted_name_lint():
    with pytest.raises(ValueError, match="dotted"):
        ParamName("undotted")


# ================================================================
#  The canonical registry
# ================================================================
def test_registry_constants_are_paramnames():
    constants = (params.TIME_STEP, params.CORIOLIS_F0,
                 params.CORIOLIS_BETA, params.STRATIFICATION_N2,
                 params.SCALING_NONLINEARITY)
    for constant in constants:
        assert isinstance(constant, ParamName)


def test_registry_canonical_strings():
    # "stepper.dt" is a confirm-at-first-use spelling (declarations.md
    # open question 5); this test pins the proposal until 2.4.
    assert params.TIME_STEP == "stepper.dt"
    assert params.CORIOLIS_F0 == "coriolis.f0"
    assert params.CORIOLIS_BETA == "coriolis.beta"
    assert params.STRATIFICATION_N2 == "stratification.n2"
    assert params.SCALING_NONLINEARITY == "scaling.nonlinearity"


def test_registry_no_default_marks():
    # a default on these would silently change the physics
    assert params.TIME_STEP.no_default is True
    assert params.STRATIFICATION_N2.no_default is True
    # identity-defaultable names stay defaultable
    assert params.CORIOLIS_F0.no_default is False
    assert params.CORIOLIS_BETA.no_default is False
    assert params.SCALING_NONLINEARITY.no_default is False


def test_registry_hints_are_nonempty():
    constants = (params.TIME_STEP, params.CORIOLIS_F0,
                 params.CORIOLIS_BETA, params.STRATIFICATION_N2,
                 params.SCALING_NONLINEARITY)
    for constant in constants:
        assert constant.hint
