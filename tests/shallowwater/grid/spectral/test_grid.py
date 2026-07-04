"""Tests for the spectral grid of the shallow water model."""

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = sw.grid.spectral.Grid(shape=(16, 16), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-7), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, time_stepper=time_stepper).setup()


# ================================================================
#  Tests
# ================================================================
def test_periodic_bounds_are_accepted():
    grid = sw.grid.spectral.Grid(
        shape=(16, 16), domain_size=(2*PI, 2*PI),
        periodic_bounds=(True, True))

    assert all(grid.periodic_bounds)


def test_non_periodic_bounds_raise():
    with pytest.raises(ValueError, match="Only periodic boundaries"):
        sw.grid.spectral.Grid(
            shape=(16, 16), domain_size=(2*PI, 2*PI),
            periodic_bounds=(True, False))


def test_omega_ignores_use_discrete(mset):
    k = mset.grid.k_mesh

    om_continuous = mset.grid.omega(k, use_discrete=False)
    om_discrete = mset.grid.omega(k, use_discrete=True)
    om_expected = sw.grid.cartesian.eigenvectors.omega(
        mset=mset, s=1, k=k, use_discrete=False)

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

        assert isinstance(z_continuous, sw.State)
        # the discrete flag is ignored on the spectral grid
        for name, field in z_continuous.fields.items():
            assert jnp.allclose(field.arr, z_discrete[name].arr)
