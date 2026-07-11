"""Tests for fridom.spatial.operators.verbs (D3b)."""
import jax.numpy as jnp
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators import verbs
from fridom.spatial.operators.base import Dispatched
from fridom.spatial.operators.registry import DispatchError


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


def test_verbs_are_interned_dispatched_singletons():
    assert verbs.diff is Dispatched("diff")
    assert verbs.interpolate is Dispatched("interpolate")
    assert verbs.integrate is Dispatched("integrate")


def test_diff_verb_resolves_against_the_field_grid(grid, mx):
    f = grid.create_field(data=jnp.arange(8.0))
    d = verbs.diff["x"](f)
    assert d.function_space.bare is mx.right


def test_interpolate_verb(grid, mx):
    f = grid.create_field(data=jnp.arange(8.0))
    h = verbs.interpolate["x"](f)
    assert h.function_space.bare is mx.right
    expected = 0.5 * (jnp.arange(8.0) + jnp.roll(jnp.arange(8.0), -1))
    assert jnp.allclose(h.data, expected)


def test_integrate_verb_resolves_the_seeded_rows(grid, mx):
    f = grid.create_field(data=jnp.full(8, 2.0))
    total = verbs.integrate["x"](f)
    assert total.function_space.bare is mx.constant
    assert jnp.allclose(total.data, 2.0)  # int over [0, 1]


def test_custom_kind_is_a_clean_dispatch_error(grid):
    f = grid.create_field()
    with pytest.raises(DispatchError, match="mykind"):
        Dispatched("mykind")["x"](f)
