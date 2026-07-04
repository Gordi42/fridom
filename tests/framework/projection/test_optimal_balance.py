"""Tests for the optimal balance projection."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr
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
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-6), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.1,
        time_stepper=time_stepper).setup()


@pytest.fixture
def z_geo(mset):
    return sw.initial_conditions.RandomGeostrophicSpectra(mset, seed=1)


def wave_production(mset, z):
    """Measure the wave energy produced by a model run started from z."""
    proj_wave = sw.projection.WaveSpectral(mset)
    model = sw.Model(mset)
    model.z = z
    model.run(runlen=2.0)
    return proj_wave(model.z - z).norm_l2()


# ================================================================
#  Tests
# ================================================================
def test_base_point_is_preserved(mset, z_geo):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=1.0, max_it=2)

    z_bal = ob(z_geo)

    z_base = proj_geo(z_geo)
    diff = (proj_geo(z_bal) - z_base).norm_l2()
    assert diff / z_base.norm_l2() < 1e-10


def test_balanced_state_produces_less_waves(mset, z_geo):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=2.0, max_it=2)

    z_naive = proj_geo(z_geo)
    z_bal = ob(z_geo)

    assert wave_production(mset, z_bal) < wave_production(mset, z_naive)


def test_ramp_functions():
    for ramp_type in ("exp", "pow", "cos", "lin"):
        ramp_func = fr.projection.OptimalBalance.get_ramp_func(ramp_type)

        # the ramp function increases from 0 to 1
        assert ramp_func(0.0) == pytest.approx(0.0)
        assert ramp_func(0.5) == pytest.approx(0.5)
        assert ramp_func(1.0) == pytest.approx(1.0)
        theta = np.linspace(0, 1, 11)
        assert (np.diff(ramp_func(theta)) >= 0).all()


def test_invalid_ramp_type_raises():
    with pytest.raises(ValueError, match="Invalid ramp type"):
        fr.projection.OptimalBalance.get_ramp_func("invalid")


def test_return_details_and_stop_criterion(mset, z_geo):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=1.0, max_it=3,
        stop_criterion=1e30, return_details=True)

    _z_bal, (iterations, errors) = ob(z_geo)

    # the huge stop criterion stops after the first iteration
    assert len(iterations) == 3
    assert errors[0] < 1e30
    assert (errors[1:] == 1).all()


def test_custom_update_parameters(mset, z_geo):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    calls = []

    def update_parameters(mset, ramped_value, mode):
        calls.append((ramped_value, mode))
        mset.tendencies.advection.scaling = ramped_value * 0.1

    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=0.25,
        update_parameters=update_parameters,
        disable_diagnostic=False, max_it=1)

    _ = ob(z_geo)

    # the custom update function is called with both ramping modes
    modes = {mode for _, mode in calls}
    assert modes == {"forward", "backward"}
    assert all(0 <= value <= 1 for value, _ in calls)


def test_increasing_error_stops_iterations(mset, z_geo, monkeypatch):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=0.25, max_it=4,
        return_details=True)

    # force an increasing error sequence
    errors = iter([0.5, 0.7, 0.1, 0.1])
    monkeypatch.setattr(
        sw.State, "norm_of_diff", lambda _self, _other: next(errors))

    _z_bal, (_its, reported_errors) = ob(z_geo)

    # the iterations stop as soon as the error increases
    assert reported_errors[0] == 0.5
    assert reported_errors[1] == 0.7
    assert (reported_errors[2:] == 1).all()


def test_reversed_time_ramping_roundtrip(mset, z_geo):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    ob = fr.projection.OptimalBalance(
        mset, base_proj=proj_geo, ramp_period=1.0)

    z_base = proj_geo(z_geo)

    # ramping backward in time to the nonlinear model and forward in
    # time back to the linear model is approximately the identity
    z_nonlinear = ob.backward_to_nonlinear(z_base)
    z_roundtrip = ob.forward_to_linear(z_nonlinear)

    assert (z_roundtrip - z_base).norm_l2() / z_base.norm_l2() < 1e-3
