"""Tests for the upwind flux function."""

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[1, 2, 3])
def n_dims(request):
    return request.param

@pytest.fixture
def shape(n_dims):
    return (3, 10, 4)[:n_dims]

@pytest.fixture
def grid(shape):
    return fr.grid.cartesian.Grid(shape=shape, domain_size=(1, 2, 3)[:len(shape)])

@pytest.fixture
def mset(grid):
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    return mset

# ================================================================
#  Tests
# ================================================================

def test_upwind_random(mset, n_dims):
    # create scalar fields for velocity and flux
    velocity = fr.ScalarField(mset)
    flux_left = fr.ScalarField(mset)
    flux_right = fr.ScalarField(mset)

    # fill with random values
    velocity.set_random(seed=12345)
    flux_left.set_random(seed=62343)
    flux_right.set_random(seed=96787)

    # create the upwind flux function and setup
    upwind = fr.modules.flux_functions.Upwind()
    upwind.setup(mset)

    # test the upwind flux function in every axis
    # compute the upwind flux
    upwind_flux = upwind.compute(flux_left, flux_right, velocity)
    # get the mask for positive and negative velocity
    pos_mask = velocity.arr >= 0
    neg_mask = velocity.arr < 0
    # check that the upwind flux is correct
    assert fr.config.ncp.allclose(
        upwind_flux.arr[pos_mask], flux_left.arr[pos_mask])
    assert fr.config.ncp.allclose(
        upwind_flux.arr[neg_mask], flux_right.arr[neg_mask])
