"""Tests for fridom.framework2.grid.fields.metadata."""
import dataclasses

import pytest

from fridom.framework2.grid.fields.metadata import FieldMetadata


def test_defaults():
    md = FieldMetadata()
    assert md.name == "unnamed"
    assert md.long_name == "Unnamed"
    assert md.units == "n/a"
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
