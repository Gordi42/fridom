"""Parameterful hydrostatic diagnostics: ekin, epot, eta and b_total."""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def make_model(n2=2.0, gravity=1.0):
    """Return a minimal linear hydrostatic model (advection=False)."""
    return hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=gravity),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


def test_ekin_is_half_the_horizontal_speed_squared():
    model = make_model()
    rng = np.random.default_rng(0)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    ekin = model.diagnostics.ekin()
    state = model.state
    centre = state["p_hyd"].function_space
    u_c = np.asarray(state["u"].to(centre).data)
    v_c = np.asarray(state["v"].to(centre).data)
    expected = 0.5 * (u_c**2 + v_c**2)
    assert np.allclose(np.asarray(ekin.data), expected)
    assert ekin.name == "ekin"


def test_ekin_is_zero_at_rest():
    model = make_model()
    ekin = model.diagnostics.ekin()
    assert float(np.abs(ekin.data).max()) == 0.0


def test_epot_is_half_b_squared_over_n2():
    n2 = 3.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(1)
    model.set_fields(b=rng.standard_normal(model.state["b"].shape))
    epot = model.diagnostics.epot()
    b = np.asarray(model.state["b"].data)
    assert np.allclose(np.asarray(epot.data), 0.5 * b**2 / n2)
    assert epot.name == "epot"


def test_epot_can_evaluate_on_a_passed_state():
    n2 = 2.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(2)
    b = model.grid.create_field(
        model.state["b"].function_space,
        data=rng.standard_normal(model.state["b"].shape))
    state = model.state.replace(b=b)
    epot = model.diagnostics.epot(state)
    assert np.allclose(np.asarray(epot.data),
                       0.5 * np.asarray(b.data)**2 / n2)


def test_eta_is_surface_pressure_over_gravity():
    gravity = 2.5
    model = make_model(gravity=gravity)
    rng = np.random.default_rng(4)
    model.set_fields(ps=rng.standard_normal(model.state["ps"].shape))
    eta = model.diagnostics.eta()
    ps = np.asarray(model.state["ps"].data)
    assert np.allclose(np.asarray(eta.data), ps / gravity)
    assert eta.name == "eta"
    assert eta.xr.attrs["units"] == "m"


def test_eta_can_evaluate_on_a_passed_state():
    gravity = 3.0
    model = make_model(gravity=gravity)
    rng = np.random.default_rng(5)
    ps = model.grid.create_field(
        model.state["ps"].function_space,
        data=rng.standard_normal(model.state["ps"].shape))
    state = model.state.replace(ps=ps)
    eta = model.diagnostics.eta(state)
    assert np.allclose(np.asarray(eta.data),
                       np.asarray(ps.data) / gravity)


# ================================================================
#  b_total: the anomaly plus the ConstantStratification background
# ================================================================
def test_b_total_adds_the_background_stratification():
    n2 = 3.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(6)
    model.set_fields(b=rng.standard_normal(model.state["b"].shape))
    total = model.diagnostics.b_total()
    b = model.state["b"]
    z = np.asarray(b.nodes("z").data)
    assert np.allclose(np.asarray(total.data),
                       np.asarray(b.data) + n2 * z)
    assert total.function_space is b.function_space
    assert total.name == "b_total"
    assert total.xr.attrs["units"] == "m/s^2"


def test_b_total_can_evaluate_on_a_passed_state():
    n2 = 2.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(7)
    b = model.grid.create_field(
        model.state["b"].function_space,
        data=rng.standard_normal(model.state["b"].shape))
    state = model.state.replace(b=b)
    total = model.diagnostics.b_total(state)
    z = np.asarray(b.nodes("z").data)
    assert np.allclose(np.asarray(total.data),
                       np.asarray(b.data) + n2 * z)


def test_b_total_is_contributed_by_the_stratification_module():
    # without a ConstantStratification there is no background to add
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    with pytest.raises(AttributeError,
                       match="no diagnostic named 'b_total'"):
        _ = model.diagnostics.b_total


def test_b_total_nondimensional_background_is_n2_z_in_physical_units():
    # Rotational frame: at rest b_total = (eps/Fr^2) z (the advection
    # carries eps, so the background gradient is N^2_eff / eps), and
    # the b_total unit row U^2/(eps H) turns it into N^2 z with the
    # Froude definition N = U/(Fr H) and z = H z
    length, speed, rossby, froude = 2.0e3, 0.5, 0.25, 0.125
    height = 1.0  # the vertical extent of make_grid
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(),
        scaling=fr.scaling.Rotational(L=length, U=speed),
        coriolis=hy.FPlaneCoriolis(rossby_number=rossby),
        buoyancy=hy.ConstantStratification(froude_number=froude),
        free_surface=hy.ExplicitFreeSurface(froude_number=0.25),
        advection=False, surface_advective_flux=False,
        time_stepper=AdamBashforth(1e-3, order=3))
    total = model.diagnostics.b_total()
    z = np.asarray(model.state["b"].nodes("z").data)
    assert np.allclose(np.asarray(total.data), rossby / froude**2 * z)
    n_freq = speed / (froude * height)
    physical = model.units.factor("b_total") * np.asarray(total.data)
    assert np.allclose(physical, n_freq**2 * height * z)
