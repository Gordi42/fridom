"""Tests for StateSignature: equality, validation, rest, repr."""
import types

import pytest

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.signature import StateSignature

from .conftest import build_state


def test_of_prognostic_from_state(state):
    sig = StateSignature.of_prognostic(state)
    assert sig.names == ("u", "v")
    assert sig.grid is state.grid
    assert sig.rest == "zero"


def test_equality_grid_identity(state, grid, other_grid):
    sig = StateSignature.of_prognostic(state)
    same = StateSignature.of_prognostic(build_state(grid))
    other = StateSignature.of_prognostic(build_state(other_grid))
    assert sig == same
    assert sig != other
    assert hash(sig) == hash(same)


def test_rest_excluded_from_equality(state):
    a = StateSignature.of_prognostic(state, rest="zero")
    b = StateSignature.of_prognostic(state, rest="pass")
    assert a == b
    assert hash(a) == hash(b)


def test_equality_other_type(state):
    sig = StateSignature.of_prognostic(state)
    assert sig.__eq__(42) is NotImplemented


def test_validate_input_accepts_matching(state):
    sig = StateSignature.of_prognostic(state)
    sig.validate_input(state)  # no raise


def test_validate_input_allows_extra_components(grid):
    two = build_state(grid, names=("u", "v"))
    three = build_state(grid, names=("u", "v", "w"))
    sig = StateSignature.of_prognostic(two)
    sig.validate_input(three)  # extra 'w' is legal (rest policy)


def test_validate_input_missing_component(grid):
    full = build_state(grid, names=("u", "v"))
    partial = build_state(grid, names=("u",))
    sig = StateSignature.of_prognostic(full)
    with pytest.raises(SignatureMismatchError, match="missing"):
        sig.validate_input(partial)


def test_validate_input_wrong_grid(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = build_state(other_grid)
    with pytest.raises(SignatureMismatchError, match="grid"):
        sig.validate_input(other)


def test_validate_input_wrong_order(grid):
    forward = build_state(grid, names=("u", "v"))
    sig = StateSignature.of_prognostic(forward)
    swapped = VectorField({"v": forward["v"], "u": forward["u"]})
    with pytest.raises(SignatureMismatchError, match="order"):
        sig.validate_input(swapped)


def test_validate_input_space_mismatch(grid, mesh, state):
    sig = StateSignature.of_prognostic(state)  # u, v on center
    staggered_u = grid.create_field(mesh.right, name="u")
    mixed = state.replace(u=staggered_u)
    with pytest.raises(SignatureMismatchError, match="space mismatch"):
        sig.validate_input(mixed)


class _FakeTable:

    """Minimal field-table surface for of_prognostic's model branch."""

    def __init__(self, state):
        self._state = state
        self.prognostic = state.component_names

    def __getitem__(self, name):
        space = self._state[name].function_space
        return types.SimpleNamespace(space=space)


def test_of_prognostic_from_model_like(state):
    model_like = type("M", (), {})()
    model_like.grid = state.grid
    model_like.field_table = _FakeTable(state)
    sig = StateSignature.of_prognostic(model_like)
    assert sig == StateSignature.of_prognostic(state)


def test_repr(state):
    sig = StateSignature.of_prognostic(state)
    text = repr(sig)
    assert "StateSignature" in text
    assert "rest='zero'" in text
