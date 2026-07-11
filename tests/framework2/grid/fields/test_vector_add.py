"""Tests for VectorField.add and the metadata re-attachment rule.

Covers phase2 grid follow-up item 6 (fields.md, 2026-07-08
amendment): ``add(**contributions)`` plus incumbent-metadata
re-attachment through ``replace``/``map``/``add``.
"""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 1.0), name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def u(grid, mx, my):
    return grid.create_field(mx.right * my.center, name="u",
                             units="m/s")


@pytest.fixture
def v(grid, mx, my):
    return grid.create_field(mx.center * my.right, name="v",
                             units="m/s")


@pytest.fixture
def b(grid, mx, my):
    return grid.create_field(mx.center * my.center, name="b",
                             units="K")


@pytest.fixture
def vec(u, v, b):
    return VectorField([u, v, b])


# ================================================================
#  add(): accumulation semantics
# ================================================================
def test_add_single_contribution(vec, grid, mx, my):
    du = grid.create_field(mx.right * my.center,
                           data=jnp.full((8, 4), 2.0))
    out = vec.add(u=du)
    assert jnp.allclose(out["u"].data, vec["u"].data + 2.0)
    assert jnp.allclose(vec["u"].data, 0.0)  # self unchanged


def test_add_multi_component_contributions(vec):
    out = vec.add(u=vec["u"] + 1.0, b=vec["b"] + 3.0)
    assert jnp.allclose(out["u"].data, vec["u"].data + 1.0)
    assert jnp.allclose(out["b"].data, vec["b"].data + 3.0)


def test_add_passes_untouched_components_through(vec, v, b):
    out = vec.add(u=vec["u"] + 1.0)
    assert out["v"] is v
    assert out["b"] is b


def test_add_unknown_component_raises(vec, u):
    with pytest.raises(KeyError, match="no components named"):
        vec.add(w=u)


def test_add_keeps_declaration_order_not_kwargs_order(vec):
    out = vec.add(b=vec["b"] + 1.0, u=vec["u"] + 1.0)
    assert out.component_names == ("u", "v", "b")
    assert tuple(out.components) == ("u", "v", "b")


def test_add_follows_the_join_rule(vec, grid, mx, my):
    # no special-casing: a mismatched contribution space raises like
    # plain scalar-field addition would
    wrong = grid.create_field(mx.center * my.center)
    with pytest.raises(SpaceMismatchError):
        vec.add(u=wrong)


# ================================================================
#  Metadata re-attachment (incumbent wins, keyed by component)
# ================================================================
def test_add_reattaches_incumbent_metadata(vec):
    # the sum itself carries default metadata (new quantity); the
    # container re-attaches the incumbent component's annotation
    out = vec.add(u=vec["u"] + 1.0)
    assert out["u"].name == "u"
    assert out["u"].metadata.units == "m/s"


def test_add_incumbent_wins_over_contribution_metadata(vec, u):
    out = vec.add(u=u.with_metadata(name="other", units="1"))
    assert out["u"].name == "u"
    assert out["u"].metadata.units == "m/s"


def test_replace_reattaches_incumbent_metadata(vec, u):
    out = vec.replace(u=u + 1.0)
    assert out["u"].name == "u"
    assert out["u"].metadata.units == "m/s"
    assert jnp.allclose(out["u"].data, u.data + 1.0)


def test_replace_incumbent_wins_over_incoming_metadata(vec, u):
    out = vec.replace(u=u.with_metadata(name="other"))
    assert out["u"].name == "u"


def test_map_reattaches_incumbent_metadata(vec):
    # f * 2.0 is bare scalar arithmetic (default metadata); map
    # restores each component's annotation
    out = vec.map(lambda f: f * 2.0)
    assert out["u"].name == "u"
    assert out["b"].metadata.units == "K"


def test_map_incumbent_wins_over_fn_metadata(vec):
    out = vec.map(lambda f: f.with_metadata(name="other"))
    assert out.component_names == ("u", "v", "b")
    assert out["v"].name == "v"
