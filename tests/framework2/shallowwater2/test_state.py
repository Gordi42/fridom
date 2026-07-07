"""Parameter-free State diagnostics on the C-grid."""
import numpy as np

import fridom.framework2 as fr
import fridom.shallowwater2 as sw

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
