"""Tests for the spectral grid of the nonhydrostatic model."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.spectral.Grid(shape=(8, 8, 8), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=1, stratification_n2=1)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    mset.tendencies.pressure_solver = \
        nh.modules.pressure_solvers.SpectralPressureSolver()
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
def test_periodic_bounds_are_accepted():
    grid = nh.grid.spectral.Grid(
        shape=(8, 8, 8), domain_size=(2*PI,)*3,
        periodic_bounds=(True, True, True))

    assert all(grid.periodic_bounds)


def test_non_periodic_bounds_raise():
    with pytest.raises(ValueError, match="Only periodic boundaries"):
        nh.grid.spectral.Grid(
            shape=(8, 8, 8), domain_size=(2*PI,)*3,
            periodic_bounds=(True, False, True))


def test_omega_ignores_use_discrete(mset):
    k = mset.grid.k_mesh

    om_continuous = mset.grid.omega(k, use_discrete=False)
    om_discrete = mset.grid.omega(k, use_discrete=True)
    om_expected = nh.grid.cartesian.eigenvectors.omega(
        s=1, f0=mset.f0, stratification_n2=mset.stratification_n2,
        dsqr=mset.dsqr, k=k, use_discrete=False)

    assert jnp.allclose(om_continuous, om_expected)
    # the discrete flag is ignored on the spectral grid
    assert jnp.allclose(om_discrete, om_continuous)


@pytest.mark.parametrize("s", [
    pytest.param(0, id="geostrophic"),
    pytest.param(1, id="wave_plus"),
    pytest.param(-1, id="wave_minus"),
])
def test_eigenvectors_ignore_use_discrete(mset, s):
    for vec in ("vec_q", "vec_p"):
        z_continuous = getattr(mset.grid, vec)(s, use_discrete=False)
        z_discrete = getattr(mset.grid, vec)(s, use_discrete=True)

        assert isinstance(z_continuous, nh.State)
        # the discrete flag is ignored on the spectral grid
        for name, field in z_continuous.fields.items():
            assert jnp.allclose(field.arr, z_discrete[name].arr)
