"""Tests for the spectral differentiation module."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
K = 2.0


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(2*PI, 2*PI))
    mset = fr.ModelSettingsBase(grid=grid)
    return mset.setup()


@pytest.fixture
def diff_module(mset):
    diff = fr.grid.cartesian.SpectralDiff()
    diff.setup(mset=mset)
    return diff


@pytest.fixture
def field(mset):
    x, _y = mset.grid.x_mesh
    f = fr.ScalarField(mset, name="f")
    f.arr = jnp.sin(K * x)
    return f.sync()


# ================================================================
#  Tests
# ================================================================
def test_spectral_derivative_is_exact(mset, diff_module, field):
    x, _y = mset.grid.x_mesh

    df = diff_module.diff(field.fft(), axis=0).ifft()

    expected = K * jnp.cos(K * x)
    assert jnp.abs(df.arr - expected).max() < 1e-12


def test_physical_field_is_transformed(mset, diff_module, field):
    # a non-spectral input is transformed to spectral space and back
    x, _y = mset.grid.x_mesh

    df = diff_module.diff(field, axis=0)

    assert not df.is_spectral
    expected = K * jnp.cos(K * x)
    assert jnp.abs(df.arr - expected).max() < 1e-12


@pytest.mark.parametrize(("bc_type", "expected_bc_type"), [
    pytest.param(fr.grid.BCType.DIRICHLET, fr.grid.BCType.NEUMANN,
                 id="dirichlet-to-neumann"),
    pytest.param(fr.grid.BCType.NEUMANN, fr.grid.BCType.DIRICHLET,
                 id="neumann-to-dirichlet"),
])
def test_bc_types_are_swapped(mset, diff_module, bc_type, expected_bc_type):
    # the derivative swaps the boundary condition type along the
    # differentiation axis and keeps the other axes
    x, _y = mset.grid.x_mesh
    f = fr.ScalarField(mset, name="f",
                       bc_types=(bc_type, fr.grid.BCType.NEUMANN))
    f.arr = jnp.sin(K * x)
    f.sync()

    df = diff_module.diff(f.fft(), axis=0)

    assert df.bc_types[0] == expected_bc_type
    assert df.bc_types[1] == fr.grid.BCType.NEUMANN


def test_wrong_grid_type_raises():
    bad_mset = MagicMock()
    bad_mset.grid = object()
    diff = fr.grid.cartesian.SpectralDiff()

    with pytest.raises(TypeError, match="spectral or cartesian grid"):
        diff.setup(mset=bad_mset)
