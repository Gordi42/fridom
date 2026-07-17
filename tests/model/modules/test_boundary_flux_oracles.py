"""BoundaryFlux physical oracles on a walled-z nonhydro2 box.

With f0 = n2 = 0 and advection off, a boundary flux is the only
tendency on the forced field, so both the pointwise source and the
stepper-integrated evolution are exact checks. A horizontally-uniform
flux keeps the velocities zero (the hydrostatic pressure adjustment
cancels the buoyancy force on w), so the projection is a no-op and the
forced field evolves freely — the buoyancy-flux and wind-stress
oracles, the global budget, the stretched-in-time scale, and the sign
convention are all pinned here. The declarations, wall-weight builder,
and taught errors live in the ``test_boundary_flux`` shard.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.modules.boundary_flux import BoundaryFlux
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 6
LX = 2 * np.pi
LZ = 1.0
DZ = LZ / N
DT = 2e-3
SIGN = {"left": 1.0, "right": -1.0}


def make_grid():
    """Return a box periodic in x, y and bounded in z on [0, LZ]."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, LZ if name == "z" else LX),
                     periodic=(name != "z"), name=name)
        for name in ("x", "y", "z")))


def make_model(*modules, stepper=None):
    """Linear nh model; f0 = n2 = 0 isolates the boundary flux."""
    kwargs = ({"dt": DT} if stepper is None
              else {"time_stepper": stepper})
    return nh.Model(
        grid=make_grid(), advection=False,
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        modules_extra=modules, **kwargs)


def x_centers():
    """Return the x cell-centre coordinates (b's own nodes along x)."""
    return (np.arange(N) + 0.5) * (LX / N)


# ================================================================
#  The pointwise source (constraints=False, exact)
# ================================================================
@pytest.mark.parametrize("side", ["right", "left"])
def test_buoyancy_flux_source_is_sign_flux_over_dz(side):
    q = 0.7
    model = make_model(BoundaryFlux("b", "z", side, flux=q))
    rng = np.random.default_rng(0)
    model.set_fields(b=rng.standard_normal(model.state["b"].data.shape))
    tend = np.asarray(model.tendency(model.state, constraints=False)["b"]
                      .data)
    expected = np.zeros_like(tend)
    idx = -1 if side == "right" else 0
    expected[:, :, idx] = SIGN[side] * q / DZ
    # the source is independent of b (a pure boundary source)
    np.testing.assert_allclose(tend, expected, atol=1e-13)


# ================================================================
#  The advance oracle: field(wall row) = sign * flux * t / Delta n
# ================================================================
@pytest.mark.parametrize("side", ["right", "left"])
def test_buoyancy_flux_advance_matches_the_integrated_flux(side):
    q, steps = 0.7, 10
    bf = BoundaryFlux("b", "z", side, flux=q)
    model = make_model(bf)
    model.advance(steps)
    t = steps * DT
    b = np.asarray(model.state["b"].data)
    idx = -1 if side == "right" else 0
    # the wall row equals the stepper-integrated q * t / Delta z
    np.testing.assert_allclose(b[:, :, idx], SIGN[side] * q * t / DZ,
                               rtol=1e-11, atol=1e-13)
    # the interior stays untouched, and the velocities stay zero
    interior = np.delete(b, idx, axis=2)
    np.testing.assert_allclose(interior, 0.0, atol=1e-12)
    for comp in ("u", "v", "w"):
        assert np.abs(np.asarray(model.state[comp].data)).max() < 1e-12


def test_global_budget_equals_flux_times_wall_area():
    q, steps = 0.5, 8
    model = make_model(BoundaryFlux("b", "z", "right", flux=q))
    model.advance(steps)
    t = steps * DT
    total = float(np.asarray(model.state["b"].integrate().data).ravel()[0])
    area = LX * LX
    np.testing.assert_allclose(total, SIGN["right"] * q * t * area,
                               rtol=1e-11, atol=1e-12)


# ================================================================
#  Wind stress on the tangential velocity u
# ================================================================
def test_wind_stress_on_u_matches_the_integrated_flux():
    tau, steps = 0.4, 10
    model = make_model(BoundaryFlux("u", "z", "right", flux=tau))
    model.advance(steps)
    t = steps * DT
    u = np.asarray(model.state["u"].data)
    np.testing.assert_allclose(u[:, :, -1], SIGN["right"] * tau * t / DZ,
                               rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.delete(u, -1, axis=2), 0.0, atol=1e-12)
    for comp in ("v", "w"):
        assert np.abs(np.asarray(model.state[comp].data)).max() < 1e-12


# ================================================================
#  Spatially varying flux F(x): pointwise and the FV budget
# ================================================================
def test_spatially_varying_flux_source_and_budget():
    def flux(x):
        return 1.0 + 0.5 * np.cos(x)

    model = make_model(BoundaryFlux("b", "z", "right", flux=flux))
    tend = np.asarray(model.tendency(model.state, constraints=False)["b"]
                      .data)
    xc = x_centers()
    # the top row samples F at b's own x nodes (a CellAvg -> CellAvg
    # broadcast, no smearing); the interior stays zero
    top = SIGN["right"] * flux(xc) / DZ
    np.testing.assert_allclose(
        tend[:, :, -1], np.broadcast_to(top[:, None], (N, N)), atol=1e-13)
    np.testing.assert_allclose(tend[:, :, :-1], 0.0, atol=1e-14)
    # the FV budget: int(tendency) dV = sign * int_wall F dA (exact)
    total = float(np.asarray(
        model.tendency(model.state, constraints=False)["b"]
        .integrate().data).ravel()[0])
    dx = LX / N
    expected = SIGN["right"] * float(np.sum(flux(xc)) * dx * LX)
    np.testing.assert_allclose(total, expected, rtol=1e-11, atol=1e-12)


# ================================================================
#  Time-dependent scale: hand-stepped forward-Euler oracle
# ================================================================
def _euler_wall_oracle(scale, q, steps):
    """Forward-Euler integral of sign * scale(n*dt) * q / Delta z."""
    total = 0.0
    for n in range(steps):
        total += DT * SIGN["right"] * float(scale.at_time(n * DT)) * q / DZ
    return total


@pytest.mark.parametrize("scale", [
    pytest.param(fr.model.Ramp(0.2, 1.0, period=12 * DT), id="ramp"),
    pytest.param(
        fr.model.TimeFunction(lambda t, w: jnp.sin(w * t), (30.0,)),
        id="sine"),
])
def test_time_dependent_scale_matches_a_hand_stepped_oracle(scale):
    q, steps = 0.7, 8
    euler = fr.model.time_steppers.AdamBashforth(DT, order=1)
    model = make_model(BoundaryFlux("b", "z", "right", flux=q,
                                    scale=scale), stepper=euler)
    model.advance(steps)
    b_top = float(np.asarray(model.state["b"].data)[0, 0, -1])
    np.testing.assert_allclose(b_top, _euler_wall_oracle(scale, q, steps),
                               rtol=1e-10, atol=1e-13)


def test_update_parameters_sweeps_the_scale_without_reassembly():
    bf = BoundaryFlux("b", "z", "right", flux=0.5)
    model = make_model(bf)
    before = np.asarray(
        model.tendency(model.state, constraints=False)["b"].data)
    model.update_parameters({bf.scale_parameter: 3.0})
    after = np.asarray(
        model.tendency(model.state, constraints=False)["b"].data)
    np.testing.assert_allclose(after, 3.0 * before, atol=1e-13)
