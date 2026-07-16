"""Parameterful hydrostatic diagnostics: ekin and epot."""
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def make_model(n2=2.0):
    """Return a minimal linear hydrostatic model (advection=False)."""
    return hy.Model(
        grid=make_grid(), dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=1.0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=3))


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
