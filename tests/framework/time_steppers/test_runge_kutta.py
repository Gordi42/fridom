"""Tests for the Runge-Kutta time steppers."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr

# ================================================================
#  Constants
# ================================================================
LAMBDA = 1.0
RK = fr.time_steppers.RKMethods


# ================================================================
#  Helpers
# ================================================================
class Decay(fr.modules.Module):

    """Linear decay tendency dz = -lambda z with exact solution exp."""

    name = "Decay"

    def _on_setup(self) -> None:
        pass

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = mz.z * (-LAMBDA)
        return mz


def make_mset(method, dt, tol=1e-6, max_dt=None):
    grid = fr.grid.cartesian.Grid(shape=(4,), domain_size=(1.0,))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var")
        return fr.VectorField(mset, field_list=[var])

    mset.state_constructor = _state_constructor
    mset.tendencies.add_module(Decay())
    mset.time_stepper = fr.time_steppers.RungeKutta(
        dt=dt, method=method, tol=tol, max_dt=max_dt)
    mset.setup()
    return mset


def make_state(mset):
    mz = fr.ModelState(mset)
    mz.z["var"].arr = jnp.ones_like(mz.z["var"].arr)
    return mz


def integration_error(method, dt, t_end=1.0):
    mset = make_mset(method, dt)
    time_stepper = mset.time_stepper
    mz = make_state(mset)
    for _ in range(round(t_end / dt)):
        mz = time_stepper.update(mz=mz)
    value = float(mz.z["var"].arr[2])
    return abs(value - np.exp(-LAMBDA * t_end))


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize(("method", "expected_order"), [
    pytest.param(RK.Euler, 1, id="euler"),
    pytest.param(RK.RK2, 2, id="rk2"),
    pytest.param(RK.RK3, 3, id="rk3"),
    pytest.param(RK.RK4, 4, id="rk4"),
    pytest.param(RK.RK4_38, 4, id="rk4-38"),
])
def test_order_of_convergence(method, expected_order):
    # integrate dz = -z to t=1 and measure the convergence order
    # against the exact solution exp(-t)
    error_coarse = integration_error(method, dt=0.1)
    error_fine = integration_error(method, dt=0.05)

    order = np.log2(error_coarse / error_fine)
    assert abs(order - expected_order) < 0.2


@pytest.mark.parametrize("method", [
    pytest.param(RK.HEUN_EULER, id="heun-euler"),
    pytest.param(RK.BOGACKI_SHAMPINE, id="bogacki-shampine"),
    pytest.param(RK.RKF45, id="rkf45"),
])
def test_adaptive_methods_control_the_error(method):
    # an adaptive method started with a too large time step must
    # reduce it until the error estimate is below the tolerance
    tol = 1e-8
    mset = make_mset(method, dt=0.5, tol=tol)
    time_stepper = mset.time_stepper
    mz = make_state(mset)

    for _ in range(5):
        mz = time_stepper.update(mz=mz)

    assert time_stepper.dt < 0.5
    value = float(mz.z["var"].arr[2])
    assert abs(value - np.exp(-LAMBDA * mz.clock.time)) < 1e-6


def test_max_dt_caps_the_time_step():
    mset = make_mset(RK.RKF45, dt=0.5, tol=1e-8, max_dt=0.01)
    time_stepper = mset.time_stepper
    assert time_stepper.dt == 0.01

    mz = make_state(mset)
    mz = time_stepper.update(mz=mz)

    # the error estimate would allow a larger step, but max_dt caps it
    assert time_stepper.dt == 0.01


def test_max_dt_setter():
    time_stepper = fr.time_steppers.RungeKutta(method=RK.RKF45)

    time_stepper.max_dt = 0.5
    assert time_stepper.max_dt == 0.5
    assert time_stepper.dt == 0.5

    time_stepper.max_dt = np.timedelta64(100, "ms")
    assert time_stepper.max_dt == pytest.approx(0.1)
    assert time_stepper.dt == pytest.approx(0.1)

    time_stepper.max_dt = None
    assert time_stepper.max_dt is None


def test_dt_from_timedelta():
    time_stepper = fr.time_steppers.RungeKutta(dt=np.timedelta64(2, "s"))
    assert time_stepper.dt == 2.0
