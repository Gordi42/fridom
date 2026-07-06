"""Tests for the nodal spaces (framework2/grid/spaces/nodal.py)."""
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.nodal import (
    NODAL_CLASS,
    Center,
    Inner,
    Left,
    NodalSpace,
    NodeSet,
    Outer,
    PointValues,
    Right,
)

N = 8


@pytest.fixture
def periodic():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 1), periodic=False, name="x")


# ================================================================
#  NodeSet marker (Wave 0)
# ================================================================
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


def test_nodal_class_mapping():
    assert NODAL_CLASS[NodeSet.CENTER] is Center
    assert NODAL_CLASS[NodeSet.LEFT] is Left
    assert NODAL_CLASS[NodeSet.RIGHT] is Right
    assert NODAL_CLASS[NodeSet.OUTER] is Outer
    assert NODAL_CLASS[NodeSet.INNER] is Inner
    assert NODAL_CLASS[NodeSet.POINTS] is PointValues


# ================================================================
#  Node-set tags
# ================================================================
@pytest.mark.parametrize(("factory", "node_set"), [
    pytest.param("center", NodeSet.CENTER, id="center"),
    pytest.param("left", NodeSet.LEFT, id="left"),
    pytest.param("right", NodeSet.RIGHT, id="right"),
    pytest.param("outer", NodeSet.OUTER, id="outer"),
    pytest.param("inner", NodeSet.INNER, id="inner"),
])
def test_node_set_tags(bounded, factory, node_set):
    space = getattr(bounded, factory)
    assert isinstance(space, NodalSpace)
    assert space.node_set is node_set


# ================================================================
#  BC-free shape table (section 3.5)
# ================================================================
@pytest.mark.parametrize(("factory", "expected"), [
    pytest.param("center", (N,), id="center"),
    pytest.param("left", (N,), id="left"),
    pytest.param("right", (N,), id="right"),
])
def test_shape_table_periodic(periodic, factory, expected):
    assert getattr(periodic, factory).shape == expected


@pytest.mark.parametrize(("factory", "expected"), [
    pytest.param("center", (N,), id="center"),
    pytest.param("left", (N,), id="left"),
    pytest.param("right", (N,), id="right"),
    pytest.param("outer", (N + 1,), id="outer"),
    pytest.param("inner", (N - 1,), id="inner"),
])
def test_shape_table_bounded(bounded, factory, expected):
    assert getattr(bounded, factory).shape == expected


# ================================================================
#  BC-constrained shapes: a constraint drops a DOF only when the
#  constrained boundary DOF is a member of the node set
# ================================================================
BOTH = (BC.DIRICHLET, BC.DIRICHLET)
LEFT_ONLY = (BC.DIRICHLET, BC.NONE)
RIGHT_ONLY = (BC.NONE, BC.DIRICHLET)


@pytest.mark.parametrize(("node_set", "bc", "expected"), [
    # Dirichlet Outer drops both boundary DOFs
    pytest.param(NodeSet.OUTER, BOTH, (N - 1,), id="outer-both"),
    pytest.param(NodeSet.OUTER, LEFT_ONLY, (N,), id="outer-left"),
    pytest.param(NodeSet.OUTER, RIGHT_ONLY, (N,), id="outer-right"),
    # Dirichlet Center keeps n DOFs: no boundary node in the set
    pytest.param(NodeSet.CENTER, BOTH, (N,), id="center-both"),
    # Inner has no boundary DOFs either
    pytest.param(NodeSet.INNER, BOTH, (N - 1,), id="inner-both"),
    # Left contains the left boundary face only
    pytest.param(NodeSet.LEFT, LEFT_ONLY, (N - 1,), id="left-left"),
    pytest.param(NodeSet.LEFT, RIGHT_ONLY, (N,), id="left-right"),
    # Right contains the right boundary face only
    pytest.param(NodeSet.RIGHT, LEFT_ONLY, (N,), id="right-left"),
    pytest.param(NodeSet.RIGHT, RIGHT_ONLY, (N - 1,), id="right-right"),
    # the rule is per non-NONE component, not Dirichlet-specific
    pytest.param(NodeSet.OUTER, (BC.NEUMANN, BC.NEUMANN), (N - 1,),
                 id="outer-neumann"),
])
def test_bc_constrained_shapes(bounded, node_set, bc, expected):
    assert bounded.nodal(node_set, bc=bc).shape == expected


def test_dirichlet_outer_is_distinct_from_inner(bounded):
    # same DOF count as BC-free Inner, yet a distinct interned space
    dirichlet_outer = bounded.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    assert dirichlet_outer.shape == bounded.inner.shape
    assert dirichlet_outer is not bounded.inner
    assert type(dirichlet_outer) is not type(bounded.inner)


# ================================================================
#  Concrete classes are dispatch tags
# ================================================================
def test_concrete_classes(bounded):
    assert type(bounded.center) is Center
    assert type(bounded.outer) is Outer
    # distinct node sets are distinct interned spaces
    assert bounded.center is not bounded.right
