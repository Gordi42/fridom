"""Tests for fridom.spatial.operators.products."""
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.products import (
    Abs,
    CollocationProduct,
    Divide,
    Power,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


@pytest.fixture
def f(grid):
    return grid.create_field(data=jnp.arange(8.0) + 1.0)


@pytest.fixture
def g(grid):
    return grid.create_field(data=jnp.full(8, 2.0))


def test_dispatch_kinds():
    assert CollocationProduct.dispatch_kind == "multiply"
    assert Divide.dispatch_kind == "divide"
    assert Power.dispatch_kind == "power"
    assert Abs.dispatch_kind == "abs"


def test_codomain_is_the_common_space(mx):
    op = CollocationProduct()
    assert op.codomain(mx.center, mx.center) is mx.center
    with pytest.raises(SpaceMismatchError, match="share one space"):
        op.codomain(mx.center, mx.right)


def test_requirements_are_halo_zero(mx):
    assert CollocationProduct().requirements(mx.center).halo == 0
    assert Abs().requirements(mx.center).halo == 0


def test_collocation_product(f, g, mx):
    h = CollocationProduct()(f, g)
    assert h.function_space.bare is mx.center
    assert jnp.allclose(h.data, f.data * 2.0)
    assert h.name == "unnamed"  # new quantity: default metadata


def test_divide(f, g):
    h = Divide()(f, g)
    assert jnp.allclose(h.data, f.data / 2.0)


def test_power(f, g):
    h = Power()(f, g)
    assert jnp.allclose(h.data, f.data ** 2.0)


def test_abs_lands_on_the_real_space(grid, mx):
    space = mx.center.as_complex()
    z = grid.create_field(space, data=jnp.full(8, 3.0 + 4.0j))
    h = Abs()(z)
    assert h.function_space.bare is mx.center
    assert jnp.allclose(h.data, jnp.full(8, 5.0))


def test_abs_codomain_on_products(mx):
    my = IntervalMesh(4, (0.0, 1.0), name="y")
    product = (mx.center * my.center).as_complex()
    assert Abs().codomain(product) is mx.center * my.center


def test_abs_is_not_bindable():
    with pytest.raises(TypeError, match="fixed signature"):
        Abs()["x"]


def test_operands_reach_products_synced(f, g):
    # storage-level products keep ghost slots consistent: the result
    # round-trips through .data without contamination
    h = CollocationProduct()(f, g)
    assert jnp.allclose(h.data, f.data * g.data)
    assert not bool(h.has_nan())
