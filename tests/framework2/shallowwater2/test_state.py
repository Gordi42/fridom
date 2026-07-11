"""Parameter-free State diagnostics on the C-grid."""
import numpy as np

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.spaces.nodal import NodeSet

from .conftest import make_grid


def _model():
    return sw.Model(grid=make_grid(),
                    time_stepper=fr.time_steppers.AdamBashforth(
                        5e-3, order=3))


def test_rel_vort_of_solid_body_rotation_sign():
    model = _model()
    # u = -sin(2pi y), v = sin(2pi x): zeta = dx v - dy u > 0 core
    model.set_fields(u=lambda x, y: -np.sin(2 * np.pi * y) + 0.0 * x,
                     v=lambda x, y: np.sin(2 * np.pi * x) + 0.0 * y)
    zeta = model.state.rel_vort
    assert zeta.name == "rel_vort"
    # a nonzero, finite vorticity field on the NE corner space
    assert np.isfinite(np.asarray(zeta.data)).all()
    assert float(np.abs(zeta.data).max()) > 0.0


def test_divergence_of_pure_divergent_flow():
    model = _model()
    # u = sin(2pi x): divergence dx u + dy v is nonzero
    model.set_fields(
        u=lambda x, y: np.sin(2 * np.pi * x) + 0.0 * y)
    div = model.state.divergence
    assert div.name == "divergence"
    assert float(np.abs(div.data).max()) > 0.0


def test_rest_state_has_zero_diagnostics():
    model = _model()  # all fields default to zero
    assert float(np.abs(model.state.rel_vort.data).max()) == 0.0
    assert float(np.abs(model.state.divergence.data).max()) == 0.0


def test_walled_rel_vort_lands_on_the_free_slip_corner_space():
    model = sw.Model(
        grid=make_grid(periodic_y=False),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    rng = np.random.default_rng(2)
    u = rng.standard_normal(model.state["u"].shape)
    v = rng.standard_normal(model.state["v"].shape)
    model.set_fields(u=u, v=v)
    zeta = model.state.rel_vort
    # the free-slip claim: the corner space adopts v's Dirichlet
    # wall tag on y (zeta = 0 at the wall); interior corners only
    my = model.grid.factors[1]
    assert (zeta.function_space.bare.factor("y")
            is my.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    # interior values are the plain centred stencils
    n = u.shape[0]
    dx = dy = 1.0 / n
    expected = ((np.roll(v, -1, axis=0) - v) / dx
                - (u[:, 1:] - u[:, :-1]) / dy)
    np.testing.assert_allclose(np.asarray(zeta.data), expected,
                               rtol=0.0, atol=1e-13)
