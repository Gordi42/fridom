"""Tests for the stage vocabulary (model/stages.py)."""
import dataclasses

import pytest

from fridom.model.stages import (
    STAGE_ATTRIBUTE,
    Stage,
    StageKind,
    self_update,
)


def body(_module, _state, _ctx):
    return {}


# ================================================================
#  StageKind
# ================================================================
def test_kind_vocabulary_is_closed():
    assert [kind.name for kind in StageKind] == [
        "SELF_UPDATE", "DIAGNOSE", "ADVANCE", "CONSTRAINT",
        "DIAGNOSTIC"]


def test_kinds_are_hashable_and_distinct():
    assert len(set(StageKind)) == 5


def test_no_open_kind_set():
    # Enum classes cannot be extended: kind declaration is the only
    # extension surface (no "insert before X" API)
    with pytest.raises(TypeError):
        class MoreKinds(StageKind):
            EXTRA = 99


# ================================================================
#  Stage construction
# ================================================================
def test_records_attributes():
    stage = Stage(kind=StageKind.ADVANCE, fn=body, name="barotropic",
                  order=2, advances=("eta", "U", "V"))
    assert stage.kind is StageKind.ADVANCE
    assert stage.fn is body
    assert stage.name == "barotropic"
    assert stage.order == 2
    assert stage.advances == ("eta", "U", "V")
    assert stage.reads == ()


def test_defaults():
    stage = Stage(kind=StageKind.CONSTRAINT, fn=body)
    assert stage.name is None
    assert stage.order == 0
    assert stage.advances == ()
    assert stage.reads == ()


def test_is_frozen():
    stage = Stage(kind=StageKind.DIAGNOSE, fn=body)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.order = 1


def test_fn_may_be_a_method_name_string():
    stage = Stage(kind=StageKind.CONSTRAINT, fn="projection")
    assert stage.fn == "projection"


def test_normalizes_name_lists_to_tuples():
    stage = Stage(kind=StageKind.ADVANCE, fn=body,
                  advances=["eta", "U"])
    assert stage.advances == ("eta", "U")
    stage = Stage(kind=StageKind.SELF_UPDATE, fn=body,
                  reads=["eta"])
    assert stage.reads == ("eta",)


def test_rejects_non_stagekind():
    with pytest.raises(TypeError, match="StageKind member"):
        Stage(kind="advance", fn=body)


def test_rejects_non_callable_non_string_fn():
    with pytest.raises(TypeError, match="callable or a method name"):
        Stage(kind=StageKind.DIAGNOSE, fn=42)


def test_advances_is_advance_or_constraint_only():
    with pytest.raises(ValueError, match="ADVANCE/CONSTRAINT-only"):
        Stage(kind=StageKind.DIAGNOSTIC, fn=body, advances=("u",))


def test_advances_accepted_on_constraint():
    # the implicit free surface's ps claim (spec 5.4 lint amendment)
    stage = Stage(kind=StageKind.CONSTRAINT, fn=body, advances=("ps",))
    assert stage.advances == ("ps",)


def test_reads_is_self_update_only():
    with pytest.raises(ValueError, match="SELF_UPDATE-only"):
        Stage(kind=StageKind.ADVANCE, fn=body, reads=("eta",))


def test_stage_has_no_cadence_field():
    # cadence= is RESERVED (CS-1), not built
    field_names = {field.name for field in dataclasses.fields(Stage)}
    assert "cadence" not in field_names
    with pytest.raises(TypeError):
        Stage(kind=StageKind.SELF_UPDATE, fn=body, cadence="step")


# ================================================================
#  The fr.self_update decorator
# ================================================================
class Geometry:

    """A module-like host with stamped self-update methods."""

    @self_update
    def self_update(self, _state, _ctx):
        return {"metric": "field"}


class ZStarGeometry:

    """The reads= spelling."""

    @self_update(reads=("eta",))
    def self_update(self, _state, _ctx):
        return {"zstar_metric": "field"}


def test_bare_form_stamps_a_self_update_stage():
    stage = getattr(Geometry.self_update, STAGE_ATTRIBUTE)
    assert isinstance(stage, Stage)
    assert stage.kind is StageKind.SELF_UPDATE
    assert stage.name == "self_update"
    assert stage.reads == ()


def test_parenthesized_form_records_reads():
    stage = getattr(ZStarGeometry.self_update, STAGE_ATTRIBUTE)
    assert stage.kind is StageKind.SELF_UPDATE
    assert stage.reads == ("eta",)


def test_fn_is_stored_unbound():
    stage = getattr(ZStarGeometry.self_update, STAGE_ATTRIBUTE)
    assert stage.fn is ZStarGeometry.__dict__["self_update"]
    assert not hasattr(stage.fn, "__self__")


def test_decorated_method_stays_plainly_callable():
    module = ZStarGeometry()
    assert module.self_update("state", "ctx") == {
        "zstar_metric": "field"}


def test_cadence_is_reserved():
    with pytest.raises(NotImplementedError, match="reserved"):
        self_update(cadence="step")


def test_cadence_rejects_even_none():
    # the keyword is accepted only to be rejected (CS-1)
    with pytest.raises(NotImplementedError, match="reserved"):
        self_update(cadence=None)


def test_cadence_rejection_points_at_the_s6_idiom():
    with pytest.raises(NotImplementedError,
                       match="S6 DIAGNOSTIC-kind stage"):
        self_update(reads=("eta",), cadence="step")


def test_docstring_carries_the_accumulation_hazard():
    # DOCSTRING-NORMATIVE (02_rules, CS-1): the hazard must be
    # documented on the decorator itself
    doc = " ".join(self_update.__doc__.split())
    assert "multi-counts" in doc
    assert "RK3" in doc
    assert "S6 DIAGNOSTIC-kind stage" in doc
    assert "cadence=" in doc
    assert "RESERVED" in doc
