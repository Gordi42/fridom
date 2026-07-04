"""Test for the equatorial wave initial condition."""
import numpy as np
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
RESOLUTION = 6  # N = 2**RESOLUTION = 64
PI = sw.config.ncp.pi

# Constants related to the Earth
EARTH_RADIUS = 6371e3
OMEGA = 2 * PI / 86400

# Define domain extent
LATITUDE_EXTENT = [-30, 30]
LONGITUDE_EXTENT = [-50, 10]  # atlantic ocean, pacific would be [-140, -80]
DEPTH = 4000
RESOLUTION_POWER = 8

# pseudo cartesian coordinates (x, y) in meters
LENGTH_X = EARTH_RADIUS * np.diff(np.deg2rad(LONGITUDE_EXTENT))[0]
LENGTH_Y = EARTH_RADIUS * np.diff(np.deg2rad(LATITUDE_EXTENT))[0]
y0 = LENGTH_Y / 2

# Compute the coriolis parameter at the equator
LATITUDE = 0
F0 = 2 * OMEGA * np.sin(np.deg2rad(LATITUDE))
BETA = 2 * OMEGA / EARTH_RADIUS * np.cos(np.deg2rad(LATITUDE))

# Compute the phase velocity of a given mode
VERTICAL_MODE = 1
STRATIFICATION_N2 = 2.5e-5
CSQR = STRATIFICATION_N2 * ( DEPTH / (np.pi * VERTICAL_MODE) ) ** 2

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture
def mset():
    # setting up the grid
    grid = sw.grid.cartesian.Grid(
        N=(2**RESOLUTION, 2**RESOLUTION),
        L=(LENGTH_X, LENGTH_Y),
        periodic_bounds=(True, False),
    )

    # setting up the model parameters
    mset = sw.ModelSettings(grid, f0=F0, beta=BETA, csqr=CSQR).setup()

    # set the correct coriolis parameter
    y_coord = mset.f_coriolis.get_mesh()[1]
    mset.f_coriolis.arr = BETA * (y_coord - LENGTH_Y / 2)

    # set the correct time step
    mset.time_stepper.dt = 0.1 * np.min(grid.dx) / np.sqrt(CSQR)

    # disable the advection term
    mset.tendencies.advection.disable()

    return mset

@pytest.fixture(params=[0, 3])
def equatorial_mode(request):
    return request.param

@pytest.fixture(params=[1, 2])
def wave_mode(request):
    return request.param

@pytest.fixture
def z_ini(mset, equatorial_mode, wave_mode):
    """Create the initial condition."""
    return sw.initial_conditions.EquatorialWave(
        mset=mset,
        longitudinal_mode=2,
        equatorial_mode=equatorial_mode,
        wave_mode=wave_mode,
    )

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
#  Tests
# ================================================================

def test_model_run(mset, z_ini):
    model = sw.Model(mset)
    model.z = z_ini
    model.run(runlen=z_ini.period)
    z_fin = model.z

    # check energy conservation
    accepted_tolerance = 1e-4
    assert compute_energy_diff(z_ini, z_fin) < accepted_tolerance

    # check if fields are approximately constant

    accepted_tolerance = 1e-1  # 10% because of the coarse resolution
    for f1, f2 in zip(z_ini, z_fin, strict=False):
        amp = f1.abs().mean().arr.item()
        rel_diff = (f1 - f2).abs().mean().arr.item() / amp
        assert rel_diff < accepted_tolerance

