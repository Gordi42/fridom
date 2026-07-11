"""Tests for the BC markers (spatial/bc.py)."""
import pytest

from fridom.spatial.bc import BC, BCStructure


# ================================================================
#  BC
# ================================================================
def test_bc_members():
    assert list(BC) == [BC.NONE, BC.DIRICHLET, BC.NEUMANN, BC.ROBIN]


def test_bc_is_hashable():
    assert len(set(BC)) == 4


def test_periodicity_is_not_a_bc_member():
    # periodicity is mesh topology (mesh.periodic), not a BC kind
    assert "PERIODIC" not in BC.__members__


# ================================================================
#  BCStructure construction
# ================================================================
def test_bcstructure_stores_components():
    structure = BCStructure((BC.DIRICHLET, BC.NEUMANN))
    assert structure.components == (BC.DIRICHLET, BC.NEUMANN)


def test_bcstructure_rejects_non_bc_components():
    with pytest.raises(TypeError, match="must be BC members"):
        BCStructure((BC.DIRICHLET, "neumann"))


# ================================================================
#  normalize
# ================================================================
def test_normalize_expands_single_bc():
    structure = BCStructure.normalize(BC.DIRICHLET, n_components=2)
    assert structure.components == (BC.DIRICHLET, BC.DIRICHLET)


def test_normalize_accepts_tuple():
    structure = BCStructure.normalize(
        (BC.NONE, BC.NEUMANN), n_components=2)
    assert structure.components == (BC.NONE, BC.NEUMANN)


def test_normalize_passes_structure_through():
    structure = BCStructure((BC.DIRICHLET, BC.NONE))
    assert BCStructure.normalize(structure, n_components=2) is structure


@pytest.mark.parametrize("spec", [
    pytest.param((BC.DIRICHLET,), id="tuple_too_short"),
    pytest.param((BC.DIRICHLET,) * 3, id="tuple_too_long"),
    pytest.param(BCStructure((BC.NONE,)), id="structure_wrong_length"),
])
def test_normalize_validates_length(spec):
    with pytest.raises(ValueError, match="boundary components"):
        BCStructure.normalize(spec, n_components=2)


# ================================================================
#  Derived properties
# ================================================================
@pytest.mark.parametrize(("components", "n_constraints", "is_free"), [
    pytest.param((BC.NONE, BC.NONE), 0, True, id="free"),
    pytest.param((BC.DIRICHLET, BC.NONE), 1, False, id="one_side"),
    pytest.param((BC.DIRICHLET, BC.NEUMANN), 2, False, id="mixed"),
    pytest.param((BC.NEUMANN, BC.NEUMANN), 2, False, id="neumann"),
])
def test_constraint_counting(components, n_constraints, is_free):
    structure = BCStructure(components)
    assert structure.n_constraints == n_constraints
    assert structure.is_free is is_free


# ================================================================
#  Value equality and hashing (unlike spaces: by value)
# ================================================================
def test_value_equality():
    a = BCStructure((BC.DIRICHLET, BC.NONE))
    b = BCStructure((BC.DIRICHLET, BC.NONE))
    c = BCStructure((BC.NONE, BC.DIRICHLET))
    assert a == b
    assert a is not b
    assert a != c
    assert a.__eq__("not a structure") is NotImplemented


def test_value_hash_usable_in_interning_keys():
    a = BCStructure((BC.DIRICHLET, BC.NONE))
    b = BCStructure((BC.DIRICHLET, BC.NONE))
    assert hash(a) == hash(b)
    registry = {("Center", a): "space"}
    assert registry[("Center", b)] == "space"


def test_repr_names_the_components():
    assert repr(BCStructure((BC.DIRICHLET, BC.NONE))) == \
        "BCStructure(DIRICHLET, NONE)"
