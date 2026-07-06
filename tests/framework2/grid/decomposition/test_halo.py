"""Tests for the negotiated halo widths (HaloSpec)."""
import dataclasses

import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec


def test_mapping_normalized_to_sorted_tuple():
    spec = HaloSpec({"y": 1, "x": 2})
    assert spec.widths == (("x", 2), ("y", 1))


def test_equality_and_hash_across_mapping_order():
    a = HaloSpec({"x": 2, "y": 1})
    b = HaloSpec({"y": 1, "x": 2})
    assert a == b
    assert hash(a) == hash(b)


def test_usable_as_dict_key():
    a = HaloSpec({"x": 1})
    b = HaloSpec({"x": 1})
    cache = {a: "compiled"}
    assert cache[b] == "compiled"


def test_zero():
    spec = HaloSpec.zero(("x", "y", "z"))
    assert spec.widths == (("x", 0), ("y", 0), ("z", 0))
    assert spec == HaloSpec({"x": 0, "y": 0, "z": 0})


def test_getitem():
    spec = HaloSpec({"x": 2, "y": 0})
    assert spec["x"] == 2
    assert spec["y"] == 0


def test_getitem_unknown_name_raises():
    spec = HaloSpec({"x": 2})
    with pytest.raises(KeyError):
        spec["z"]


def test_negative_width_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        HaloSpec({"x": -1})


def test_frozen():
    spec = HaloSpec({"x": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.widths = ()


def test_grow_returns_new_spec():
    spec = HaloSpec({"x": 1, "y": 2})
    grown = spec.grow("x", 2)
    assert grown["x"] == 3
    assert grown["y"] == 2
    # the original is unchanged (frozen value semantics)
    assert spec["x"] == 1


def test_grow_chains_add():
    # sequential un-synced applications add their widths
    spec = HaloSpec.zero(("x",))
    grown = spec.grow("x", 1).grow("x", 2).grow("x", 3)
    assert grown["x"] == 6


def test_grow_by_zero_is_identity_value():
    spec = HaloSpec({"x": 2})
    assert spec.grow("x", 0) == spec


def test_grow_unknown_name_raises():
    spec = HaloSpec({"x": 1})
    with pytest.raises(KeyError):
        spec.grow("z", 1)


def test_grow_negative_rejected():
    spec = HaloSpec({"x": 1})
    with pytest.raises(ValueError, match="grow amount"):
        spec.grow("x", -1)


def test_merge_max_pointwise():
    # parallel expression branches take the pointwise maximum
    a = HaloSpec({"x": 3, "y": 1})
    b = HaloSpec({"x": 1, "y": 2})
    assert a.merge_max(b) == HaloSpec({"x": 3, "y": 2})
    assert b.merge_max(a) == HaloSpec({"x": 3, "y": 2})


def test_merge_max_union_of_names():
    # a name missing from one spec counts as width 0
    a = HaloSpec({"x": 2})
    b = HaloSpec({"y": 1})
    merged = a.merge_max(b)
    assert merged == HaloSpec({"x": 2, "y": 1})


def test_accumulation_rules_compose():
    # grow along a chain, then max against a parallel branch
    chain = HaloSpec.zero(("x", "y")).grow("x", 2).grow("x", 1)
    branch = HaloSpec({"x": 1, "y": 2})
    assert chain.merge_max(branch) == HaloSpec({"x": 3, "y": 2})
