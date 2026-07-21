"""Shared fixtures for the shallowwater2 wave-6 smoke tests."""
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

N = 16
DT = 5e-3


def make_grid(n=N, *, periodic_x=True, periodic_y=True):
    """Return a tiny square grid (walled along non-periodic axes)."""
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my))


def make_model(grid=None, *, csqr=1.0, rossby_number=0.2, f0=1.0,
               dt=DT, order=3, advection=True, coriolis=True,
               **kwargs):
    """Build a shallow-water model on the scaling surface.

    The (csqr, rossby_number, f0) knobs keep the historical test
    parametrization and spell the TODAY-PARITY nondimensional model:
    GravityWave scaling, core froude_number = rossby_number with the
    depth ratio csqr, Coriolis Ro = rossby_number / f0 (the live
    epsilon/Ro ratio then equals f0 exactly for dyadic pins).
    """
    if grid is None:
        grid = make_grid()
    if coriolis is True:
        coriolis = sw.modules.FPlaneCoriolis(
            rossby_number=rossby_number / f0)
    elif coriolis is False:
        coriolis = None
    return sw.Model(
        grid=grid,
        core=sw.Core(froude_number=rossby_number, depth=csqr),
        scaling=fr.scaling.GravityWave(),
        coriolis=coriolis,
        advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=order),
        **kwargs)


def gaussian_bump(amp=0.1, sigma=0.12):
    """Return a centred Gaussian pressure perturbation (rest velocities)."""
    def p(x, y):
        return amp * np.exp(
            -((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))
    return p


def total_energy(model):
    """Compute a centre-sampled total-energy proxy of the state.

    ``E = integral[ 0.5 Ro^2 p_full (u^2 + v^2) + 0.5 p_full^2 ]``
    with ``p_full = c^2 + Ro p`` — the old model's ekin + epot,
    evaluated on cell centres.
    """
    state = model.state
    ro = float(model.parameters[fr.model.params.SCALING_NONLINEARITY])
    c = state["csqr"]
    p = state["p"]
    centre = p.function_space
    u_c = state["u"].to(centre)
    v_c = state["v"].to(centre)
    p_full = c.to(centre) + ro * p
    ekin = 0.5 * ro ** 2 * p_full * (u_c * u_c + v_c * v_c)
    epot = 0.5 * (p_full * p_full)
    return float((ekin + epot).integrate().data.ravel()[0])


@pytest.fixture
def grid():
    """Return a shared tiny periodic grid (fresh per test)."""
    return make_grid()
