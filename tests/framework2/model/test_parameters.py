"""Tests for fridom.framework2.model.parameters (declaration surface)."""
import dataclasses

import pytest

from fridom.framework2.model.parameters import (
    REQUIRED,
    USE_PROVIDED,
    Param,
    ParameterDeclaration,
    ParameterReference,
)
from fridom.framework2.model.params import SCALING_ROSSBY, ParamName


# ================================================================
#  The REQUIRED / USE_PROVIDED sentinels
# ================================================================
def test_sentinels_are_distinct():
    assert REQUIRED is not USE_PROVIDED
    assert REQUIRED != USE_PROVIDED


def test_sentinel_reprs_are_helpful():
    assert "REQUIRED" in repr(REQUIRED)
    assert "no default" in repr(REQUIRED)
    assert "USE_PROVIDED" in repr(USE_PROVIDED)
    assert "binding table" in repr(USE_PROVIDED)


# ================================================================
#  ParameterDeclaration
# ================================================================
def test_declaration_construction_and_defaults():
    decl = ParameterDeclaration("coriolis.f0", attr="f0")
    assert decl.name == "coriolis.f0"
    assert decl.attr == "f0"
    assert decl.units == "n/a"
    assert decl.doc == ""


def test_declaration_accepts_paramname():
    decl = ParameterDeclaration(SCALING_ROSSBY, attr="rossby_input")
    # the declaration keys the binding table by the plain string
    assert decl.name == "scaling.rossby"


def test_declaration_is_frozen():
    decl = ParameterDeclaration("coriolis.f0", attr="f0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        decl.attr = "other"


def test_declaration_value_semantics():
    a = ParameterDeclaration("coriolis.f0", attr="f0", units="1/s")
    b = ParameterDeclaration("coriolis.f0", attr="f0", units="1/s")
    c = ParameterDeclaration("coriolis.f0", attr="beta", units="1/s")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


# ================================================================
#  ParameterReference
# ================================================================
def test_reference_defaults_to_required():
    ref = ParameterReference("stratification.n2")
    assert ref.name == "stratification.n2"
    assert ref.hint == ""
    assert ref.default is REQUIRED


def test_reference_with_identity_default():
    ref = ParameterReference("scaling.rossby", default=1.0)
    assert ref.default == 1.0


def test_reference_is_a_namedtuple():
    ref = ParameterReference("scaling.rossby", "a hint", 1.0)
    name, hint, default = ref
    assert (name, hint, default) == ("scaling.rossby", "a hint", 1.0)
    assert isinstance(ref, tuple)


def test_reference_value_semantics():
    a = ParameterReference("scaling.rossby", default=1.0)
    b = ParameterReference(ParamName("scaling.rossby"), default=1.0)
    assert a == b
    assert hash(a) == hash(b)


# ================================================================
#  Param (reference-valued constructor slots)
# ================================================================
def test_param_defaults_to_required():
    slot = Param("scaling.rossby")
    assert slot.name == "scaling.rossby"
    assert slot.default is REQUIRED


def test_param_with_identity_default():
    # the flagship consumer spelling: generic advection stays
    # Ro-ignorant via scaling=fr.Param("scaling.rossby", default=1.0)
    slot = Param(SCALING_ROSSBY, default=1.0)
    assert slot.name == "scaling.rossby"
    assert slot.default == 1.0


def test_param_is_frozen():
    slot = Param("scaling.rossby", default=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.default = 2.0


def test_param_value_semantics():
    a = Param("scaling.rossby", default=1.0)
    b = Param("scaling.rossby", default=1.0)
    assert a == b
    assert hash(a) == hash(b)
    assert a != Param("scaling.rossby")
