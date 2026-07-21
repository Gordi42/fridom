"""nonhydro2 surface forcing: WindStress and SurfaceBuoyancyFlux.

These wrappers own the oceanographic sign conventions (BF-D4) on top of
the generic BoundaryFlux. The sign conventions are pinned AGAINST the
raw BoundaryFlux spelling (the wrappers must reduce to it exactly), and
the taught errors are shown to be inherited from the shared validation.
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.modules.boundary_flux import BoundaryFlux
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 6
LX = 2 * np.pi
LZ = 1.0
DZ = LZ / N


def make_grid(*, walled="z"):
    """Return a box periodic except along ``walled`` (bounded [0, LZ])."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, LZ if name == walled else LX),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z")))


def make_model(*modules, walled="z"):
    """Linear nh model; f0 = n2 = 0 isolates the surface forcing."""
    return nh.Model(
        grid=make_grid(walled=walled),
        core=nh.Core(),
        time_stepper=AdamBashforth(2e-3, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=False,
        modules_extra=modules)


def advanced(*modules, steps=6, field="b"):
    """Advance a model with the given modules and return the field data."""
    model = make_model(*modules)
    model.advance(steps)
    return np.asarray(model.state[field].data)


# ================================================================
#  WindStress: sign convention against the raw BoundaryFlux spelling
# ================================================================
def test_windstress_tau_x_equals_raw_boundary_flux_minus_tau():
    tau = 0.4
    got = advanced(nh.WindStress(tau_x=tau), field="u")
    raw = advanced(BoundaryFlux("u", "z", "right", flux=-tau), field="u")
    np.testing.assert_array_equal(got, raw)


def test_windstress_tau_y_equals_raw_boundary_flux_minus_tau():
    tau = 0.3
    got = advanced(nh.WindStress(tau_y=tau), field="v")
    raw = advanced(BoundaryFlux("v", "z", "right", flux=-tau), field="v")
    np.testing.assert_array_equal(got, raw)


def test_windstress_spatially_varying_tau_matches_raw():
    got = advanced(nh.WindStress(tau_x=lambda y: 1.0 + np.sin(y)),
                   field="u")
    raw = advanced(
        BoundaryFlux("u", "z", "right", flux=lambda y: -(1.0 + np.sin(y))),
        field="u")
    np.testing.assert_array_equal(got, raw)


def test_positive_tau_x_accelerates_the_surface_in_plus_x():
    u = advanced(nh.WindStress(tau_x=0.5), steps=5, field="u")
    # the top cell accelerates in +x; the interior stays at rest
    assert u[0, 0, -1] > 0
    np.testing.assert_allclose(u[:, :, :-1], 0.0, atol=1e-12)


def test_windstress_forces_u_and_v_from_one_module():
    ws = nh.WindStress(tau_x=0.2, tau_y=-0.3)
    assert (ws.coord, ws.side) == ("z", "right")
    terms = ws.tendency_terms()
    assert len(terms) == 1
    assert terms[0].advances == ("u", "v")


def test_windstress_publishes_a_side_qualified_scale():
    ws = nh.WindStress(tau_x=0.2)
    model = make_model(ws)
    assert ws.scale_parameter == "wind_stress.z_right.scale"
    assert "wind_stress.z_right.scale" in model.parameters


def test_windstress_update_parameters_sweeps_the_scale():
    ws = nh.WindStress(tau_x=0.3)
    model = make_model(ws)
    before = np.asarray(
        model.tendency(model.state, constraints=False)["u"].data)
    model.update_parameters({ws.scale_parameter: 2.0})
    after = np.asarray(
        model.tendency(model.state, constraints=False)["u"].data)
    np.testing.assert_allclose(after, 2.0 * before, atol=1e-13)


# ================================================================
#  SurfaceBuoyancyFlux: a positive q is a wall gain
# ================================================================
def test_surface_buoyancy_flux_equals_raw_boundary_flux_minus_q():
    q = 0.5
    got = advanced(nh.SurfaceBuoyancyFlux(q))
    raw = advanced(BoundaryFlux("b", "z", "right", flux=-q))
    np.testing.assert_array_equal(got, raw)


def test_surface_buoyancy_flux_callable_q_matches_raw():
    got = advanced(nh.SurfaceBuoyancyFlux(lambda x: 1.0 + 0.5 * np.cos(x)))
    raw = advanced(BoundaryFlux(
        "b", "z", "right", flux=lambda x: -(1.0 + 0.5 * np.cos(x))))
    np.testing.assert_array_equal(got, raw)


@pytest.mark.parametrize(("side", "idx"), [("right", -1), ("left", 0)])
def test_positive_q_is_a_buoyancy_gain_at_the_wall(side, idx):
    q, steps = 0.6, 8
    b = advanced(nh.SurfaceBuoyancyFlux(q, side=side), steps=steps)
    # a gain: the wall row grows to +q * t / Delta z
    np.testing.assert_allclose(b[:, :, idx], q * steps * 2e-3 / DZ,
                               rtol=1e-11, atol=1e-13)


def test_surface_buoyancy_flux_publishes_its_own_scale_name():
    sb = nh.SurfaceBuoyancyFlux(0.5)
    assert sb.scale_parameter == "surface_buoyancy_flux.z_right.scale"
    model = make_model(sb)
    before = np.asarray(
        model.tendency(model.state, constraints=False)["b"].data)
    model.update_parameters({sb.scale_parameter: 3.0})
    after = np.asarray(
        model.tendency(model.state, constraints=False)["b"].data)
    np.testing.assert_allclose(after, 3.0 * before, atol=1e-13)


# ================================================================
#  Combined run
# ================================================================
def test_combined_wind_and_buoyancy_stays_finite():
    model = make_model(nh.WindStress(tau_x=0.3, tau_y=-0.2),
                       nh.SurfaceBuoyancyFlux(0.4))
    model.advance(5)
    assert not model.panicked
    for comp in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(model.state[comp].data)).all()


# ================================================================
#  Opposite-wall coexistence and same-wall collision (side-qualified)
# ================================================================
def test_top_and_bottom_buoyancy_flux_coexist():
    # heating the top and cooling the bottom (Rayleigh-Benard): the
    # side-qualified scale names and AUXILIARY field names must not clash
    model = make_model(nh.SurfaceBuoyancyFlux(0.5, side="right"),
                       nh.SurfaceBuoyancyFlux(-0.3, side="left"))
    params = model.parameters
    assert "surface_buoyancy_flux.z_right.scale" in params
    assert "surface_buoyancy_flux.z_left.scale" in params
    model.advance(4)
    assert not model.panicked
    b = np.asarray(model.state["b"].data)
    assert b[0, 0, -1] > 0   # gain at the top
    assert b[0, 0, 0] < 0    # loss at the bottom


def test_top_and_bottom_wind_stress_coexist():
    model = make_model(nh.WindStress(tau_x=0.3, side="right"),
                       nh.WindStress(tau_x=0.3, side="left"))
    params = model.parameters
    assert "wind_stress.z_right.scale" in params
    assert "wind_stress.z_left.scale" in params
    model.advance(4)
    assert not model.panicked


def test_two_buoyancy_fluxes_on_the_same_wall_collide():
    with pytest.raises(Exception, match=r"surface_buoyancy_flux|bflux_b"):
        make_model(nh.SurfaceBuoyancyFlux(0.5),
                   nh.SurfaceBuoyancyFlux(0.3))


def test_two_wind_stresses_on_the_same_wall_collide():
    with pytest.raises(Exception, match=r"wind_stress|windstress_"):
        make_model(nh.WindStress(tau_x=0.1), nh.WindStress(tau_x=0.2))


# ================================================================
#  Construction-time validation
# ================================================================
def test_windstress_side_and_stress_validation():
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        nh.WindStress(side="top")
    with pytest.raises(TypeError, match="tau_x must be a number"):
        nh.WindStress(tau_x="strong")
    with pytest.raises(TypeError, match="tau_y must be a number"):
        nh.WindStress(tau_y="gentle")


def test_surface_buoyancy_flux_side_and_q_validation():
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        nh.SurfaceBuoyancyFlux(0.5, side="top")
    with pytest.raises(TypeError, match="q must be a number"):
        nh.SurfaceBuoyancyFlux("warm")


# ================================================================
#  Taught errors inherited from the shared validation
# ================================================================
def test_windstress_on_a_wall_normal_velocity_is_rejected():
    # an x-wall makes u the wall-normal velocity (staggered along x)
    with pytest.raises(ValueError, match="staggered along 'x'"):
        make_model(nh.WindStress(coord="x"), walled="x")


def test_windstress_on_a_periodic_coord_is_rejected():
    with pytest.raises(ValueError, match="is periodic"):
        make_model(nh.WindStress(coord="x"))


def test_surface_buoyancy_flux_on_a_periodic_coord_is_rejected():
    with pytest.raises(ValueError, match="is periodic"):
        make_model(nh.SurfaceBuoyancyFlux(0.5, coord="x"))


def test_windstress_stress_callable_naming_the_normal_is_rejected():
    with pytest.raises(ValueError, match="names the normal coordinate"):
        make_model(nh.WindStress(tau_x=lambda z: z))
