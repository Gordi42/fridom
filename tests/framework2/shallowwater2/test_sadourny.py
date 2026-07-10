"""Sadourny advection: the csqr-field behaviour and conservation."""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.shallowwater2 import params as sw_params

from .conftest import (
    N,
    gaussian_bump,
    make_grid,
    make_model,
    total_energy,
)

CSQR = 0.7


# ================================================================
#  Walled-grid helpers: the exactly-conserved discrete energy
# ================================================================
def walled_model(*, periodic_x=True, f0=0.0, ro=0.4, dt=2e-3):
    """Return a nonlinear walled model (y walls; x optional)."""
    grid = make_grid(periodic_x=periodic_x, periodic_y=False)
    return make_model(grid, csqr=CSQR, rossby_number=ro, f0=f0,
                      advection=True, dt=dt)


def set_random(model, seed=11, amp=1.0):
    """Fill the prognostics with random data.

    Walls are structural: v carries interior faces only, so any
    data satisfies u.n = 0.
    """
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=amp * rng.standard_normal(model.state["u"].shape),
        v=amp * rng.standard_normal(model.state["v"].shape),
        p=0.3 * amp * rng.standard_normal(model.state["p"].shape))


def h_energy_terms(model):
    """Per-term chain rule of the conserved discrete energy.

    ``E = sum 1/2 hbar^x u^2 + 1/2 hbar^y v^2 + 1/2 p^2`` with
    ``h = c^2 + Ro p`` (measure-weighted sums); the exact invariant
    of gravity + Sadourny advection on periodic AND walled grids
    (see the module docstring of shallowwater2/modules/sadourny.py).
    Returns the five dE/dt contributions as floats.
    """
    z = model.state
    dz = model.tendency(z)
    ro = float(model.parameters[fr.params.SCALING_ROSSBY])
    u, v, p = z["u"], z["v"], z["p"]
    du, dv, dp = dz["u"], dz["v"], dz["p"]
    h = z["csqr"].to(p) + ro * p
    terms = (
        u * du * h.to(u),
        0.5 * ro * (u * u) * dp.to(u),
        v * dv * h.to(v),
        0.5 * ro * (v * v) * dp.to(v),
        p * dp,
    )
    return [float(t.integrate().data.ravel()[0]) for t in terms]


def h_energy(model):
    """Evaluate the conserved discrete energy functional."""
    z = model.state
    ro = float(model.parameters[fr.params.SCALING_ROSSBY])
    u, v, p = z["u"], z["v"], z["p"]
    h = z["csqr"].to(p) + ro * p
    parts = (0.5 * u * u * h.to(u), 0.5 * v * v * h.to(v),
             0.5 * p * p)
    return sum(float(t.integrate().data.ravel()[0]) for t in parts)


# ================================================================
#  The csqr-FIELD delta (§8.8): Sadourny reads state["csqr"]
# ================================================================
def test_csqr_is_a_state_field_not_a_scalar():
    model = make_model(csqr=1.5)
    c = model.state["csqr"]
    # a one-DOF Profile() field (constant depth) that broadcasts to the
    # nodal join in the tendency terms — still a field, not a scalar
    assert isinstance(c, fr.grid.ScalarField)
    assert c.function_space.bare is (
        fr.Profile().resolve(model.grid))
    np.testing.assert_allclose(np.asarray(c.data), 1.5)


def test_csqr_scalar_is_published_for_host_reads():
    model = make_model(csqr=2.25)
    assert float(model.parameters[sw_params.CSQR]) == 2.25


def test_advection_changes_the_solution_vs_linear():
    grid = make_grid()
    nonlin = make_model(grid, rossby_number=0.5, advection=True)
    linear = make_model(grid, rossby_number=0.5, advection=False)
    nonlin.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    linear.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    nonlin.advance(40)
    linear.advance(40)
    # the nonlinear advection leaves a measurable imprint
    diff = np.abs(np.asarray(nonlin.state["p"].data)
                  - np.asarray(linear.state["p"].data)).max()
    assert diff > 1e-4


# ================================================================
#  Conservation: mass exact, energy/enstrophy bounded
# ================================================================
def test_mass_exact_under_full_dynamics():
    model = make_model(rossby_number=0.5)
    model.set_fields(p=gaussian_bump(amp=0.15),
                     v=lambda x, y: 0.1 * np.cos(2 * np.pi * x) + 0.0 * y)
    mass0 = float(model.state["p"].integrate().data.ravel()[0])
    model.advance(50)
    mass1 = float(model.state["p"].integrate().data.ravel()[0])
    assert abs(mass1 - mass0) < 1e-12


def test_energy_bounded_with_a_balanced_start():
    model = make_model(rossby_number=0.3, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.08),
                     u=lambda x, y: 0.05 * np.sin(2 * np.pi * y) + 0.0 * x,
                     v=lambda x, y: 0.05 * np.sin(2 * np.pi * x) + 0.0 * y)
    e0 = total_energy(model)
    peak = 0.0
    for _ in range(20):
        model.advance(5)
        peak = max(peak, abs(total_energy(model) - e0) / e0)
    assert peak < 5e-3


def test_potential_enstrophy_stays_bounded():
    model = make_model(rossby_number=0.4, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.1),
                     u=lambda x, y: 0.1 * np.sin(2 * np.pi * y) + 0.0 * x)

    def enstrophy(m):
        # 0.5 * integral zeta^2 (NE corner) — a bounded invariant
        z = m.state.rel_vort
        return float((0.5 * z * z).integrate().data.ravel()[0])

    q0 = enstrophy(model)
    model.advance(80)
    q1 = enstrophy(model)
    assert np.isfinite(q1)
    assert q1 < 5.0 * (q0 + 1e-9)   # no runaway enstrophy growth


# ================================================================
#  Walled grids: free-slip Sadourny (channel and doubly-walled)
# ================================================================
@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
@pytest.mark.parametrize("seed", [3, 11])
def test_walled_semi_discrete_energy_rate_is_machine_zero(
        periodic_x, seed):
    # gravity + Sadourny advection conserve E = sum 1/2 hbar^x u^2
    # + 1/2 hbar^y v^2 + 1/2 p^2 exactly (semi-discrete) for ANY
    # state honouring u.n = 0 (structural on the walled spaces);
    # the vorticity-flux exchange is antisymmetric for any finite q
    # and the wall values consumed by the fills are exact zeros.
    model = walled_model(periodic_x=periodic_x, f0=0.0)
    set_random(model, seed=seed)
    terms = h_energy_terms(model)
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-13


@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
def test_walled_mass_rate_is_machine_zero(periodic_x):
    # impermeability closes the thickness flux at the walls exactly
    model = walled_model(periodic_x=periodic_x, f0=1.0)
    set_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14


def test_walled_interior_tendency_matches_periodic_stencils():
    # far from the walls (> stencil reach of 2 cells) the walled
    # tendencies are the periodic stencil values bit for bit
    rng = np.random.default_rng(9)
    u = rng.standard_normal((N, N))
    v = rng.standard_normal((N, N))
    v[:, -1] = 0.0                    # the wall face of the torus
    p = 0.3 * rng.standard_normal((N, N))

    periodic = make_model(make_grid(), csqr=CSQR, rossby_number=0.4,
                          f0=1.0, advection=True)
    walled = walled_model(f0=1.0)
    periodic.set_fields(u=u, v=v, p=p)
    walled.set_fields(u=u, v=v[:, :-1], p=p)

    dzp = periodic.tendency(periodic.state)
    dzw = walled.tendency(walled.state)
    inner = slice(3, N - 3)
    for name in ("u", "p"):
        np.testing.assert_allclose(
            np.asarray(dzw[name].data)[:, inner],
            np.asarray(dzp[name].data)[:, inner],
            rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(dzw["v"].data)[:, inner],
        np.asarray(dzp["v"].data)[:, inner],
        rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="channel"),
    pytest.param(False, id="double-walled"),
])
def test_walled_nonlinear_run_is_stable(periodic_x):
    # O(600) steps of the full nonlinear walled model: no NaN and a
    # bounded drift of the exactly-conserved discrete energy (AB3
    # time error + the split-Coriolis commutator, both small)
    model = walled_model(periodic_x=periodic_x, f0=1.0, ro=0.3,
                         dt=2e-3)
    model.set_fields(
        p=gaussian_bump(amp=0.08),
        u=lambda x, y: 0.05 * np.sin(2 * np.pi * y) + 0.0 * x)
    e0 = h_energy(model)
    peak = 0.0
    for _ in range(20):
        model.advance(30)             # 600 steps
        for name in ("u", "v", "p"):
            assert not bool(model.state[name].has_nan())
        peak = max(peak, abs(h_energy(model) - e0) / e0)
    assert peak < 5e-3


# ================================================================
#  Variable depth: Sadourny reads the csqr(y) FIELD (verified)
# ================================================================
def varying_walled_model(*, f0=0.0, ro=0.4):
    """Build a variable-depth walled channel (csqr(y) field)."""
    coriolis = sw.modules.FPlaneCoriolis(
        f0=f0, metric_weight="csqr")
    return sw.Model(
        grid=make_grid(periodic_y=False),
        csqr=lambda y: 1.0 + 0.5 * np.sin(np.pi * y),
        rossby_number=ro, coriolis=coriolis, advection=True,
        time_stepper=fr.time_steppers.AdamBashforth(2e-3, order=3))


@pytest.mark.parametrize("seed", [3, 11])
def test_varying_depth_energy_rate_is_machine_zero(seed):
    # the h-weighted invariant (h = c^2(y) + Ro p) holds for the
    # variable depth too: the scheme reads the csqr FIELD (the §8.8
    # fix), so gravity + advection telescope exactly
    model = varying_walled_model(f0=0.0)
    set_random(model, seed=seed)
    terms = h_energy_terms(model)
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-13


def test_varying_depth_mass_rate_is_machine_zero():
    model = varying_walled_model(f0=1.0)
    set_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14
