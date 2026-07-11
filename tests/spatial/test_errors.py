"""Tests for the grid-cluster error types (spatial/errors.py)."""
import pytest

from fridom.spatial.errors import (
    GridMismatchError,
    SpaceMismatchError,
)


# ================================================================
#  SpaceMismatchError
# ================================================================
def test_space_mismatch_is_a_type_error():
    assert issubclass(SpaceMismatchError, TypeError)
    with pytest.raises(TypeError, match="cannot add"):
        raise SpaceMismatchError("cannot add fields")


def test_space_mismatch_stores_attributes():
    left, right = object(), object()
    error = SpaceMismatchError(
        "cannot add fields: x: Right vs Center (y agrees)",
        left=left,
        right=right,
        operation="+",
        mismatched_names=("x",),
    )
    assert error.left is left
    assert error.right is right
    assert error.operation == "+"
    assert error.mismatched_names == ("x",)
    assert "Right vs Center" in str(error)


def test_space_mismatch_defaults():
    error = SpaceMismatchError("boom")
    assert error.left is None
    assert error.right is None
    assert error.operation is None
    assert error.mismatched_names == ()


def test_space_mismatch_normalizes_mismatched_names():
    error = SpaceMismatchError("boom", mismatched_names=["x", "z"])
    assert error.mismatched_names == ("x", "z")


def test_space_mismatch_keyword_only_attributes():
    with pytest.raises(TypeError):
        SpaceMismatchError("boom", object(), object())


# ================================================================
#  GridMismatchError
# ================================================================
def test_grid_mismatch_is_a_type_error():
    assert issubclass(GridMismatchError, TypeError)
    with pytest.raises(TypeError, match="different grids"):
        raise GridMismatchError("fields live on different grids")


def test_grid_mismatch_stores_attributes():
    left, right = object(), object()
    error = GridMismatchError(
        "fields live on different grids",
        left=left,
        right=right,
        operation="*",
    )
    assert error.left is left
    assert error.right is right
    assert error.operation == "*"


def test_grid_mismatch_defaults():
    error = GridMismatchError("boom")
    assert error.left is None
    assert error.right is None
    assert error.operation is None


def test_grid_mismatch_keyword_only_attributes():
    with pytest.raises(TypeError):
        GridMismatchError("boom", object(), object())


def test_the_two_errors_are_distinct():
    assert not issubclass(GridMismatchError, SpaceMismatchError)
    assert not issubclass(SpaceMismatchError, GridMismatchError)
