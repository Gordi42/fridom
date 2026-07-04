"""Tests for the model settings of the shallow water model."""

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw


@pytest.fixture
def mset():
    grid = sw.grid.cartesian.Grid((16, 16), (1.0, 1.0))
    return sw.ModelSettings(grid, f0=1.0, csqr=2.0).setup()


def test_setup(mset):
    assert jnp.allclose(mset.f_coriolis.arr, 1.0)
    assert jnp.allclose(mset.csqr_field.arr, 2.0)


def test_parameters(mset):
    parameters = mset.parameters
    assert parameters["coriolis parameter f0"] == "1.0 s⁻¹"
    assert parameters["Phase velocity c²"] == "2.0 m²s⁻²"
    assert "Rossby number Ro" in parameters


def test_beta_plane(mset):
    mset.beta = 2.0
    assert mset.beta == 2.0
    # the coriolis field is updated: f = f0 + beta * y
    y = mset.f_coriolis.get_mesh()[1]
    assert jnp.allclose(mset.f_coriolis.arr, 1.0 + 2.0 * y)


def test_f_coriolis_setter(mset):
    field = sw.ScalarField(mset, name="my f")
    mset.f_coriolis = field
    assert mset.f_coriolis is field

    with pytest.raises(TypeError, match="must be a ScalarField"):
        mset.f_coriolis = 42


def test_csqr_setter_updates_field(mset):
    mset.csqr = 5.0
    assert mset.csqr == 5.0
    assert jnp.allclose(mset.csqr_field.arr, 5.0)


def test_csqr_field_setter(mset):
    field = sw.ScalarField(mset, name="my csqr")
    mset.csqr_field = field
    assert mset.csqr_field is field

    with pytest.raises(TypeError, match="must be a ScalarField"):
        mset.csqr_field = 42


def test_rossby_number_scales_advection(mset):
    mset.rossby_number = 0.25
    assert mset.rossby_number == 0.25
    assert mset.tendencies.advection.scaling == 0.25
