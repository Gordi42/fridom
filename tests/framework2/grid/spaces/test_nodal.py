"""Tests for the NodeSet marker (framework2/grid/spaces/nodal.py)."""
import pytest

from fridom.framework2.grid.spaces.nodal import NodeSet


def test_node_set_members():
    assert list(NodeSet) == [
        NodeSet.CENTER,
        NodeSet.LEFT,
        NodeSet.RIGHT,
        NodeSet.OUTER,
        NodeSet.INNER,
        NodeSet.POINTS,
    ]


def test_node_set_is_hashable():
    # NodeSet members are interning-key components
    assert len(set(NodeSet)) == 6
    registry = {(NodeSet.CENTER, "REAL"): "space"}
    assert registry[(NodeSet.CENTER, "REAL")] == "space"


@pytest.mark.parametrize("member", list(NodeSet))
def test_node_set_identity(member):
    assert NodeSet[member.name] is member


def test_no_string_keyed_variant():
    # deliberately no string aliasing: strings are not members
    assert all(not isinstance(member.value, str) for member in NodeSet)
