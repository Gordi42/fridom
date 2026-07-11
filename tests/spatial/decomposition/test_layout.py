"""Tests for the device layout descriptor."""
import dataclasses

import pytest

from fridom.spatial.decomposition.layout import Layout


def test_mapping_normalized_to_sorted_tuple():
    layout = Layout({"y": "py", "x": "px"})
    assert layout.device_axes == (("x", "px"), ("y", "py"))


def test_equality_and_hash_across_mapping_order():
    a = Layout({"x": "px", "y": "py"})
    b = Layout({"y": "py", "x": "px"})
    assert a == b
    assert hash(a) == hash(b)


def test_usable_as_dict_key():
    a = Layout({"x": "px"})
    b = Layout({"x": "px"})
    cache = {a: "sharding"}
    assert cache[b] == "sharding"


def test_empty_layout_is_all_local():
    layout = Layout({})
    assert layout.device_axes == ()
    assert layout.is_local("x")
    assert layout.is_local("anything")


def test_is_local():
    layout = Layout({"x": "px"})
    assert not layout.is_local("x")
    assert layout.is_local("y")


def test_duplicate_device_axis_rejected():
    with pytest.raises(ValueError, match="at most one"):
        Layout({"x": "p0", "y": "p0"})


def test_distinct_layouts_differ():
    assert Layout({"x": "px"}) != Layout({"x": "py"})
    assert Layout({"x": "px"}) != Layout({})


def test_frozen():
    layout = Layout({"x": "px"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.device_axes = ()
