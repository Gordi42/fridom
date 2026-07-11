import jax.numpy as jnp
import pytest

import fridom.nonhydro as nh


@pytest.fixture(params=[(16, 16, 16), (16, 13, 9)],
                ids=["16x16x16", "16x13x9"])
def N(request):
    return request.param

@pytest.fixture(params=[(1.0, 1.0, 1.0), (1.0, 2.0, 3.0)],
                ids=["1x1x1", "1x2x3"])
def L(request):
    return request.param

@pytest.fixture
def grid_ini(N, L):
    return nh.grid.cartesian.Grid(N, L)

@pytest.fixture(params=[1e-4, 2], ids=["f=1e-4", "f=2"])
def f(request):
    return request.param

@pytest.fixture(params=[1e-4, 2],
                ids=["stratification_n2=1e-4", "stratification_n2=2"])
def N2(request):
    return request.param

def test_model_settings(grid_ini, f, N2):
    mset = nh.ModelSettings(grid_ini, f0=f, stratification_n2=N2)
    mset.setup()
    assert mset.grid == grid_ini
    assert jnp.allclose(mset.f_coriolis.arr, f)
    assert mset.stratification_n2 == N2


@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid((8, 8, 8), (1.0, 1.0, 1.0))
    return nh.ModelSettings(grid, f0=1.0, stratification_n2=4.0).setup()


def test_parameters(mset):
    parameters = mset.parameters
    assert parameters["coriolis parameter f0"] == "1.0 1/s"
    assert parameters["Stratification N²"] == "4.0 1/s^2"
    assert "Rossby number Ro" in parameters
    assert "Aspect ratio dsqr" in parameters


def test_beta_plane(mset):
    mset.beta = 2.0
    assert mset.beta == 2.0
    # the coriolis field is updated: f = f0 + beta * y
    y = mset.f_coriolis.get_mesh()[1]
    assert jnp.allclose(mset.f_coriolis.arr, 1.0 + 2.0 * y)


def test_f_coriolis_setter(mset):
    field = nh.ScalarField(mset, name="my f")
    mset.f_coriolis = field
    assert mset.f_coriolis is field

    with pytest.raises(TypeError, match="must be a ScalarField"):
        mset.f_coriolis = 42


def test_stratification_field_setter(mset):
    field = nh.ScalarField(mset, name="my N2")
    mset.stratification_n2_field = field
    assert mset.stratification_n2_field is field

    with pytest.raises(TypeError, match="must be a ScalarField"):
        mset.stratification_n2_field = 42


def test_stratification_setter_updates_field(mset):
    mset.stratification_n2 = 9.0
    assert mset.stratification_n2 == 9.0
    assert jnp.allclose(mset.stratification_n2_field.arr, 9.0)


def test_rossby_number_scales_advection(mset):
    mset.rossby_number = 0.25
    assert mset.rossby_number == 0.25
    assert mset.tendencies.advection.scaling == 0.25


def test_dsqr_setter(mset):
    mset.dsqr = 0.5
    assert mset.dsqr == 0.5
