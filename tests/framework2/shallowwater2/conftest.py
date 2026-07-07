"""Shared fixtures for the shallowwater2 wave-6 smoke tests."""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw

N = 16
DT = 5e-3


def make_grid(n=N):
    """Return a tiny doubly-periodic square grid."""
    mx = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                     name="y")
    return fr.grid.Grid((mx, my))


def make_model(grid=None, *, csqr=1.0, rossby_number=0.2, f0=1.0,
               dt=DT, order=3, advection=True, **kwargs):
    """Build a shallow-water model through the preset factory."""
    if grid is None:
        grid = make_grid()
    return sw.Model(
        grid=grid, csqr=csqr, rossby_number=rossby_number,
        coriolis=sw.modules.FPlaneCoriolis(f0=f0),
        advection=advection,
        time_stepper=fr.time_steppers.AdamBashforth(dt, order=order),
        **kwargs)


def gaussian_bump(amp=0.1, sigma=0.12):
    """Return a centred Gaussian height perturbation (rest velocities)."""
    def h(x, y):
        return amp * np.exp(
            -((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))
    return h


def total_energy(model):
    """Compute a centre-sampled total-energy proxy of the state.

    ``E = integral[ 0.5 Ro^2 h_full (u^2 + v^2) + 0.5 h_full^2 ]``
    with ``h_full = c^2 + Ro h`` — the old model's ekin + epot,
    evaluated on cell centres.
    """
    state = model.state
    ro = float(model.parameters[fr.params.SCALING_ROSSBY])
    c = state["csqr"]
    h = state["h"]
    centre = h.function_space
    u_c = state["u"].to(centre)
    v_c = state["v"].to(centre)
    h_full = c + ro * h
    ekin = 0.5 * ro ** 2 * h_full * (u_c * u_c + v_c * v_c)
    epot = 0.5 * (h_full * h_full)
    return float((ekin + epot).integrate().data.ravel()[0])


@pytest.fixture
def grid():
    """Return a shared tiny periodic grid (fresh per test)."""
    return make_grid()
