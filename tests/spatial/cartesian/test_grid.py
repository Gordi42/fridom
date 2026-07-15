"""Tests for fridom.spatial.cartesian.grid."""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.cartesian.grid import Grid
from fridom.spatial.grid import Grid as BaseGrid


# ================================================================
#  Construction: shapes, names, periodicity
# ================================================================
def test_is_a_grid_subclass():
    grid = Grid((8, 8), ((0.0, 1.0), (0.0, 2.0)))
    assert isinstance(grid, BaseGrid)


def test_default_names_and_extents():
    grid = Grid((8, 8), ((0.0, 1.0), (0.0, 2.0)))
    mx, my = grid.factors
    assert grid.names == ("x", "y")
    assert mx.extent == (0.0, 1.0)
    assert my.extent == (0.0, 2.0)


def test_periodic_defaults_to_true_for_every_axis():
    grid = Grid((8, 8), ((0.0, 1.0), (0.0, 1.0)))
    assert all(f.periodic for f in grid.factors)


def test_periodic_tuple_is_per_axis():
    grid = Grid((8, 8), ((0.0, 1.0), (0.0, 1.0)),
                periodic=(True, False))
    assert [f.periodic for f in grid.factors] == [True, False]


@pytest.mark.parametrize(
    ("ndim", "expected"),
    [(1, ("x",)), (2, ("x", "y")), (3, ("x", "y", "z"))])
def test_default_names_by_dimension(ndim, expected):
    grid = Grid((4,) * ndim, ((0.0, 1.0),) * ndim)
    assert grid.names == expected


def test_custom_names():
    grid = Grid((4, 4), ((0.0, 1.0), (0.0, 1.0)), names=("a", "b"))
    assert grid.names == ("a", "b")


def test_builds_a_usable_grid():
    grid = Grid((8, 8), ((0.0, 1.0), (0.0, 1.0)))
    mx, my = grid.factors
    field = grid.create_field(
        (mx.center * my.center).bare,
        init=lambda x, y: jnp.sin(2 * np.pi * x) + 0.0 * y)
    assert field.data.shape[0] > 0


# ================================================================
#  Construction errors (taught)
# ================================================================
def test_extent_length_must_match_shape():
    with pytest.raises(ValueError, match="extent has"):
        Grid((8, 8), ((0.0, 1.0),))


def test_periodic_length_must_match_shape():
    with pytest.raises(ValueError, match="periodic has"):
        Grid((8, 8), ((0.0, 1.0), (0.0, 1.0)), periodic=(True,))


def test_names_length_must_match_shape():
    with pytest.raises(ValueError, match="names has"):
        Grid((8, 8), ((0.0, 1.0), (0.0, 1.0)), names=("x",))


def test_names_required_beyond_three_dimensions():
    with pytest.raises(ValueError, match="names is required"):
        Grid((4, 4, 4, 4), ((0.0, 1.0),) * 4)
