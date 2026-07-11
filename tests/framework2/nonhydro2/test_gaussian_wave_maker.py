"""The Gaussian wave maker: envelope sampling, oscillation, errors.

On a zero state with advection off, every other nonhydro term
vanishes, so the (unconstrained) tendency of the forced variable IS
the wave-maker source — the envelope and the sinusoid are exact
checks against the mask sampled at the variable's own nodes.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.model.errors import MissingFieldError
from fridom.nonhydro2.modules.gaussian_wave_maker import (
    GaussianWaveMaker,
)

N = 8
LENGTH = 2 * np.pi
DT = 1e-3
FREQ = 2.0
AMP = 0.3


def make_grid(walled=()):
    return Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=name not in walled,
                     name=name)
        for name in ("x", "y", "z")))


def make_model(maker, walled=()):
    return nh.Model(grid=make_grid(walled), dt=DT, advection=False,
                    modules_extra=(maker,))


def nodes(model, variable, axis):
    """Return the variable's own node positions along ``axis``."""
    space = model.state[variable].function_space
    return np.asarray(
        model.grid.evaluation_nodes(space, axis).data)


def gaussian(x, pos, width):
    return np.exp(-((x - pos) ** 2) / width**2)


# ================================================================
#  Construction-time validation
# ================================================================
def test_position_and_width_must_share_keys():
    with pytest.raises(ValueError, match="same"):
        GaussianWaveMaker({"x": 1.0}, {"y": 1.0}, FREQ, AMP)
    with pytest.raises(TypeError, match="non-empty field name"):
        GaussianWaveMaker({}, {}, FREQ, AMP, variable="")


def test_the_forced_variable_defaults_to_u():
    assert GaussianWaveMaker({}, {}, FREQ, AMP).variable == "u"


# ================================================================
#  Assembly-time validation (taught errors)
# ================================================================
def test_unknown_envelope_coordinate_is_rejected_at_bind():
    maker = GaussianWaveMaker({"q": 1.0}, {"q": 1.0}, FREQ, AMP)
    with pytest.raises(ValueError, match="grid does not have"):
        make_model(maker)


def test_unknown_variable_is_a_taught_assembly_error():
    maker = GaussianWaveMaker({}, {}, FREQ, AMP, variable="uu")
    with pytest.raises(MissingFieldError, match="forced variable"):
        make_model(maker)


def test_diagnostic_variable_is_rejected_at_bind():
    maker = GaussianWaveMaker({}, {}, FREQ, AMP, variable="p")
    with pytest.raises(ValueError, match="only PROGNOSTIC"):
        make_model(maker)


def test_non_collocated_unknown_variable_space_is_rejected():
    class StaggeredTracer(fr.model.Module):
        field_declarations = (
            fr.model.FieldDeclaration("q", space=fr.spatial.Staggered("x"),
                                long_name="Staggered tracer"),
        )

    maker = GaussianWaveMaker({}, {}, FREQ, AMP, variable="q")
    with pytest.raises(ValueError, match="samples its envelope"):
        fr.model.Model(
            grid=make_grid(),
            modules=(nh.DynamicalCore(), nh.FPlaneCoriolis(),
                     nh.ConstantStratification(), StaggeredTracer(),
                     maker),
            time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))


# ================================================================
#  The forcing: envelope at the variable's own nodes, sinusoid
# ================================================================
def test_forcing_vanishes_at_time_zero():
    maker = GaussianWaveMaker({"x": np.pi}, {"x": 0.5}, FREQ, AMP)
    model = make_model(maker)
    tendency = model.tendency(model.state, t=0.0, constraints=False)
    assert np.abs(np.asarray(tendency["u"].data)).max() == 0.0


def test_envelope_is_sampled_at_the_staggered_u_nodes():
    maker = GaussianWaveMaker(
        {"x": np.pi, "y": 2.0}, {"x": 0.5, "y": 1.0}, FREQ, AMP)
    model = make_model(maker)
    # sin(2 pi f t) = 1 at t = 1 / (4 f)
    tendency = model.tendency(model.state, t=1.0 / (4.0 * FREQ),
                              constraints=False)
    x = nodes(model, "u", "x")
    y = nodes(model, "u", "y")
    expected = AMP * gaussian(x, np.pi, 0.5) * gaussian(y, 2.0, 1.0)
    got = np.asarray(tendency["u"].data)
    np.testing.assert_allclose(
        got, np.broadcast_to(expected, got.shape), atol=1e-15)
    # the staggered u faces differ from the cell centers
    centers = (np.arange(N) + 0.5) * (LENGTH / N)
    assert not np.allclose(np.unique(x), centers)


def test_collocated_buoyancy_forcing_uses_the_b_nodes():
    maker = GaussianWaveMaker({"z": np.pi}, {"z": 0.7}, FREQ, AMP,
                              variable="b")
    model = make_model(maker)
    tendency = model.tendency(model.state, t=1.0 / (4.0 * FREQ),
                              constraints=False)
    z = nodes(model, "b", "z")
    got = np.asarray(tendency["b"].data)
    np.testing.assert_allclose(
        got, np.broadcast_to(AMP * gaussian(z, np.pi, 0.7),
                             got.shape), atol=1e-15)


def test_forcing_oscillates_at_the_configured_frequency():
    maker = GaussianWaveMaker({"x": np.pi}, {"x": 0.5}, FREQ, AMP)
    model = make_model(maker)
    t1, t2 = 0.03, 0.09
    f1 = np.asarray(model.tendency(
        model.state, t=t1, constraints=False)["u"].data)
    f2 = np.asarray(model.tendency(
        model.state, t=t2, constraints=False)["u"].data)
    ratio = (np.sin(2 * np.pi * FREQ * t2)
             / np.sin(2 * np.pi * FREQ * t1))
    np.testing.assert_allclose(f2, ratio * f1, atol=1e-14)


def test_walled_vertical_grid_forces_w_at_its_wall_tagged_faces():
    maker = GaussianWaveMaker({"z": np.pi}, {"z": 0.7}, FREQ, AMP,
                              variable="w")
    model = make_model(maker, walled=("z",))
    tendency = model.tendency(model.state, t=1.0 / (4.0 * FREQ),
                              constraints=False)
    z = nodes(model, "w", "z")
    got = np.asarray(tendency["w"].data)
    np.testing.assert_allclose(
        got, np.broadcast_to(AMP * gaussian(z, np.pi, 0.7),
                             got.shape), atol=1e-15)


# ================================================================
#  Provided parameters (update without re-assembly)
# ================================================================
def test_update_parameters_scales_amplitude_and_frequency():
    maker = GaussianWaveMaker({"x": np.pi}, {"x": 0.5}, FREQ, AMP)
    model = make_model(maker)
    t = 1.0 / (4.0 * FREQ)
    before = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    model.update_parameters({"wavemaker.u.amplitude": 2 * AMP})
    doubled = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    np.testing.assert_allclose(doubled, 2.0 * before, atol=1e-14)
    # halving the frequency turns sin(pi/2) into sin(pi/4)
    model.update_parameters({"wavemaker.u.frequency": FREQ / 2})
    slowed = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    np.testing.assert_allclose(
        slowed, 2.0 * before * np.sin(np.pi / 4), atol=1e-14)


def test_energy_enters_a_running_model():
    maker = GaussianWaveMaker({"x": np.pi}, {"x": 0.5}, FREQ, AMP)
    model = make_model(maker)
    model.advance(50)
    u = np.asarray(model.state["u"].data)
    assert np.isfinite(u).all()
    assert np.abs(u).max() > 0.0
