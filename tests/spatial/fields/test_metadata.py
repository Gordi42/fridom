"""Tests for fridom.spatial.fields.metadata."""
import dataclasses

import pytest

from fridom.spatial.fields.metadata import FieldMetadata


def test_defaults():
    md = FieldMetadata()
    assert md.name == "unnamed"
    assert md.long_name == "Unnamed"
    assert md.units == "unknown"
    assert md.physical_units == "unknown"
    assert md.nondimensional is False
    assert md.nc_attrs == ()


def test_create_normalizes_mapping_to_sorted_tuple():
    md = FieldMetadata.create(
        name="u", units="m/s",
        nc_attrs={"zeta": "1", "axis": "X"})
    assert md.name == "u"
    assert md.units == "m/s"
    assert md.nc_attrs == (("axis", "X"), ("zeta", "1"))


def test_create_defaults_match_plain_constructor():
    assert FieldMetadata.create() == FieldMetadata()


def test_create_accepts_tuple_nc_attrs():
    md = FieldMetadata.create(nc_attrs=(("b", "2"), ("a", "1")))
    assert md.nc_attrs == (("a", "1"), ("b", "2"))


def test_hashable_and_value_equal():
    a = FieldMetadata.create(name="u", nc_attrs={"k": "v"})
    b = FieldMetadata.create(name="u", nc_attrs={"k": "v"})
    assert a == b
    assert hash(a) == hash(b)
    assert a != FieldMetadata.create(name="v")


def test_frozen():
    md = FieldMetadata()
    with pytest.raises(dataclasses.FrozenInstanceError):
        md.name = "u"


def test_replace():
    md = FieldMetadata.create(name="u", units="m/s")
    new = md.replace(name="v")
    assert new.name == "v"
    assert new.units == "m/s"
    assert md.name == "u"


def test_replace_normalizes_nc_attrs_mapping():
    md = FieldMetadata()
    new = md.replace(nc_attrs={"b": "2", "a": "1"})
    assert new.nc_attrs == (("a", "1"), ("b", "2"))


# ================================================================
#  Scaling-aware unit rendering
# ================================================================
def test_units_renders_dimensionless_when_nondimensional():
    md = FieldMetadata.create(name="u", units="m/s",
                              nondimensional=True)
    assert md.units == "1"
    assert md.physical_units == "m/s"
    assert md.units_declared is True


def test_units_keeps_the_physical_string_when_dimensional():
    md = FieldMetadata.create(name="u", units="m/s")
    assert md.units == "m/s"
    assert md.units_declared is True


def test_undeclared_units_stay_unknown_under_either_scaling():
    # nondimensionalizing an unknown quantity yields an unknown
    # quantity, not a dimensionless one
    for nondimensional in (False, True):
        md = FieldMetadata.create(name="q",
                                  nondimensional=nondimensional)
        assert md.units == "unknown"
        assert md.units_declared is False


def test_replace_accepts_units_as_sugar_for_physical_units():
    md = FieldMetadata.create(name="u", units="m/s",
                              nondimensional=True)
    assert md.replace(units="m^2/s^2").physical_units == "m^2/s^2"
    assert md.replace(units="m^2/s^2").units == "1"


def test_replace_rejects_both_unit_spellings():
    md = FieldMetadata.create(name="u", units="m/s")
    with pytest.raises(ValueError, match="sugar for physical_units"):
        md.replace(units="m", physical_units="m")


def test_the_nondimensional_flag_is_a_functional_update():
    md = FieldMetadata.create(name="u", units="m/s")
    stamped = md.replace(nondimensional=True)
    assert stamped.units == "1"
    assert md.units == "m/s"
