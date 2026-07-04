"""Test for the random geostrophic spectra initial condition."""

from functools import partial

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
RESOLUTION = 6  # N = 2**RESOLUTION - 1 = 63
PI = jnp.pi

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture(params=[
    pytest.param((True, True), id="double_periodic"),
    pytest.param((False, False), id="non_periodic"),
    pytest.param((True, False), id="periodic_x"),
    pytest.param((False, True), id="periodic_y"),
])
def periodic_bounds(request):
    return request.param

@pytest.fixture
def mset(periodic_bounds):
    # setting up the grid
    grid = sw.grid.cartesian.Grid(
        shape=(2**RESOLUTION - 1, 2**RESOLUTION - 1),
        domain_size=(2*PI, 2*PI),
        periodic_bounds=periodic_bounds,
    )

    # setting up the model parameters
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-RESOLUTION), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.3, time_stepper=time_stepper,
    ).setup()

@pytest.fixture
def z_ini(mset):
    """Create the initial condition."""
    spectra = sw.initial_conditions.geostrophic_energy_spectrum
    return sw.initial_conditions.RandomGeostrophicSpectra(
        mset, seed=0, spectral_energy_density=partial(spectra, k0=3))

# ================================================================
#  Helper functions
# ================================================================
def compute_energy_diff(z_ini, z_final):
    """Compute the relative energy change."""
    initial_energy = z_ini.etot.integrate().arr.item()
    final_energy = z_final.etot.integrate().arr.item()
    energy_change = final_energy - initial_energy
    return energy_change / initial_energy

# ================================================================
#  Main example
# ================================================================
def test_main_example():
    # setting up the grid
    grid = sw.grid.cartesian.Grid(
        shape=(2**RESOLUTION - 1, 2**RESOLUTION - 1),
        domain_size=(2*PI, 2*PI),
        periodic_bounds=(True, True),
    )

    # setting up the model parameters
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-RESOLUTION), order=3)
    mset = sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.3, time_stepper=time_stepper,
    ).setup()

    # creating the initial condition
    spectra = sw.initial_conditions.geostrophic_energy_spectrum
    z_ini = sw.initial_conditions.RandomGeostrophicSpectra(
        mset, seed=0, spectral_energy_density=partial(spectra, k0=3))


    model = sw.Model(mset)
    model.z = z_ini
    model.run(runlen=2*PI)

    # compute the energy change
    initial_energy = z_ini.etot.integrate().arr.item()
    final_energy = model.z.etot.integrate().arr.item()
    energy_change = final_energy - initial_energy
    relative_energy_change = energy_change / initial_energy

    accepted_tolerance = 1e-4

    assert relative_energy_change < accepted_tolerance

# ================================================================
#  Tests
# ================================================================

def test_linear_model(mset, z_ini):
    mset.tendencies.advection.disable()

    model = sw.Model(mset)
    model.z = z_ini
    model.run(runlen=2*PI)

    accepted_tolerance = 1e-10

    # check that there is no energy change
    assert compute_energy_diff(z_ini, model.z) < accepted_tolerance
    # check that the fields have not changed
    for field in z_ini - model.z:
        assert field.integrate().arr.item() < accepted_tolerance


def test_nonlinear_model(mset, z_ini):
    mset.tendencies.advection.enable()

    model = sw.Model(mset)
    model.z = z_ini
    model.run(runlen=2*PI)

    accepted_tolerance = 1e-4

    assert compute_energy_diff(z_ini, model.z) < accepted_tolerance
