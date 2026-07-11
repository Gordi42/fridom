"""Tests for the biased interpolation base class."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(16.0, 16.0))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.halo = 3
    return mset.setup()


@pytest.fixture
def module(mset):
    module = fr.grid.cartesian.UpwindInterpolation(order=2)
    return module.setup(mset=mset)


@pytest.fixture
def field(mset):
    return fr.ScalarField(mset, name="f")


# ================================================================
#  Tests
# ================================================================
def test_same_position_returns_field(module, field):
    bias = jnp.ones_like(field.arr)
    res = module.interpolate(field, bias, field.position)
    assert res is field


def test_multiple_axes_raise(module, field):
    bias = jnp.ones_like(field.arr)
    destination = field.position.shift(0).shift(1)
    with pytest.raises(ValueError, match="exactly one axis"):
        module.interpolate(field, bias, destination)


def test_bias_field_with_wrong_position_raises(mset, module, field):
    # the bias field is at the cell center, but the destination is at
    # the cell face
    bias = fr.ScalarField(mset, name="bias")
    with pytest.raises(ValueError, match="same position as the"):
        module.interpolate(field, bias, field.position.shift(0))


def test_bias_scalar_field(mset, module, field):
    destination = field.position.shift(0)
    bias = fr.ScalarField(mset, name="bias", position=destination)
    bias.arr = jnp.ones_like(bias.arr)

    res = module.interpolate(field, bias, destination)

    assert res.position == destination


def test_axes_without_extent_are_skipped(mset, module):
    f = fr.ScalarField(mset, name="f", topo=(False, True))
    bias = jnp.ones_like(f.arr)
    res = module.interpolate(f, bias, f.position.shift(0))
    assert res is f
