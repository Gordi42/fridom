"""Tests for the model state class."""
import pytest
import xarray as xr

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
def add_constructors(mset):
    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var")
        return fr.VectorField(mset, field_list=[var])

    def _diagnostic_state_constructor() -> fr.VectorField:
        diag = fr.ScalarField(mset, name="diag")
        return fr.VectorField(mset, field_list=[diag])

    mset.state_constructor = _state_constructor
    mset.diagnostic_state_constructor = _diagnostic_state_constructor
    return mset


@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid=grid)
    return add_constructors(mset).setup()


@pytest.fixture
def mz(mset):
    return fr.ModelState(mset)


# ================================================================
#  Tests
# ================================================================
def test_reset(mz):
    mz.z += 1.0
    mz.clock.tick(1.0)
    mz.reset()
    assert float(mz.z["var"].arr.max()) == 0.0
    assert mz.clock.time == 0.0


def test_setters_convert_to_physical(mset, mz):
    # on a physical grid, spectral inputs are transformed back
    mz.z = mset.state_constructor().fft()
    assert not mz.z.is_spectral

    mz.z_diag = mz.z_diag.fft()
    assert not mz.z_diag.is_spectral

    mz.dz = mset.state_constructor().fft()
    assert not mz.dz.is_spectral

    # the tendency can be reset with None
    mz.dz = None
    assert mz.dz is None


def test_setters_with_empty_state():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid=grid).setup()

    # the default state constructor creates empty vectors, which are
    # set without any space conversion
    mz = fr.ModelState(mset)
    assert mz.z.vector_dim == 0
    assert mz.z_diag.vector_dim == 0
    assert mz.dz.vector_dim == 0


def test_panicked_flag(mz):
    assert not mz.panicked
    mz.panicked = True
    assert mz.panicked


def test_xr(mz):
    dataset = mz.xr
    assert isinstance(dataset, xr.Dataset)
    assert "var" in dataset


def test_xrs(mz):
    dataset = mz.xrs[:4, :4]
    assert isinstance(dataset, xr.Dataset)
    assert dataset["var"].shape == (4, 4)
