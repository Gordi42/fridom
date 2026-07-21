"""Named analytic initial conditions: single_wave and jet."""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.initial_conditions import _vertical_extent
from fridom.hydrostatic.state import State
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=6, depth=2.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid=None):
    """Return a minimal linear hydrostatic model (advection=False)."""
    if grid is None:
        grid = make_grid()
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


# ================================================================
#  single_wave: a baroclinic buoyancy seed
# ================================================================
def test_single_wave_seeds_only_the_buoyancy():
    depth, nz = 2.0, 6
    model = make_model(make_grid(nz=nz, depth=depth))
    kx, ky, m, amp = 1, 1, 1, 1e-3
    state = hy.single_wave(
        model, wavenumbers=(kx, ky, m), amplitude=amp)
    assert isinstance(state, State)
    # u, v, ps stay at rest; only b is perturbed
    assert float(np.abs(state["u"].data).max()) == 0.0
    assert float(np.abs(state["v"].data).max()) == 0.0
    assert float(np.abs(state["ps"].data).max()) == 0.0
    assert float(np.abs(state["b"].data).max()) > 0.0

    # the analytic form A sin(2pi kx x) sin(2pi ky y) cos(m pi z / H)
    sb = model.state["b"].function_space

    def b_fn(x, y, z):
        return (amp * np.sin(2 * np.pi * kx * x)
                * np.sin(2 * np.pi * ky * y)
                * np.cos(m * np.pi * z / depth))

    expected = model.grid.create_field(sb, init=b_fn)
    assert np.allclose(np.asarray(state["b"].data),
                       np.asarray(expected.data))

    # assignable to the model
    model.set_state(state)
    assert np.allclose(np.asarray(model.state["b"].data),
                       np.asarray(expected.data))


def test_single_wave_amplitude_scales_linearly():
    model = make_model()
    small = hy.single_wave(model, amplitude=1e-3)
    big = hy.single_wave(model, amplitude=2e-3)
    assert np.allclose(2.0 * np.asarray(small["b"].data),
                       np.asarray(big["b"].data))


# ================================================================
#  jet: a barotropic zonal jet with a balancing surface pressure
# ================================================================
def test_jet_seeds_u_and_ps_only():
    model = make_model()
    state = hy.jet(model, amplitude=0.1, width=0.15)
    assert isinstance(state, State)
    assert float(np.abs(state["u"].data).max()) > 0.0
    assert float(np.abs(state["ps"].data).max()) > 0.0
    assert float(np.abs(state["v"].data).max()) == 0.0
    assert float(np.abs(state["b"].data).max()) == 0.0
    # peak velocity approaches the requested amplitude (the nearest
    # cell node sits a half cell off the channel centre, so it is a
    # hair below the analytic 0.1 peak)
    peak = float(np.abs(state["u"].data).max())
    assert 0.0 < peak <= 0.1
    assert np.isclose(peak, 0.1, atol=1.5e-2)
    model.set_state(state)
    assert float(np.abs(model.state["ps"].data).max()) > 0.0


# ================================================================
#  _vertical_extent: the missing-coordinate error branch
# ================================================================
def test_vertical_extent_returns_the_mesh_bounds():
    model = make_model(make_grid(depth=2.0))
    assert _vertical_extent(model, "z") == (0.0, 2.0)


def test_vertical_extent_raises_on_a_missing_coordinate():
    model = make_model()
    with pytest.raises(ValueError, match="coordinate"):
        _vertical_extent(model, "q")
