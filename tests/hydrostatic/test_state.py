"""The hydrostatic ``State`` vocabulary: accessors and diagnostics."""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.state import State
from fridom.spatial.errors import MissingComponentError

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def make_model():
    """Return a minimal linear hydrostatic model (advection=False)."""
    return hy.Model(
        grid=make_grid(), dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=1.0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=3))


# ================================================================
#  The state is a hy.State and carries the vocabulary
# ================================================================
def test_model_state_is_the_hydrostatic_vocabulary():
    model = make_model()
    assert isinstance(model.state, State)
    # order: u, v, w, p_hyd, [f_coriolis], b, ps
    assert model.state.component_names[:4] == ("u", "v", "w", "p_hyd")
    assert "b" in model.state.component_names
    assert "ps" in model.state.component_names


def test_vocabulary_accessors_return_components():
    state = make_model().state
    assert state.u is state["u"]
    assert state.v is state["v"]
    assert state.w is state["w"]
    assert state.b is state["b"]
    assert state.ps is state["ps"]


# ================================================================
#  Missing components raise a hinted MissingComponentError
# ================================================================
@pytest.mark.parametrize(
    ("accessor", "match"),
    [
        pytest.param("u", "hydrostatic core", id="u"),
        pytest.param("v", "hydrostatic core", id="v"),
        pytest.param("w", "hydrostatic core", id="w"),
        pytest.param("b", "stratification", id="b"),
        pytest.param("ps", "free-surface", id="ps"),
    ],
)
def test_missing_component_accessor_raises_hinted(accessor, match):
    grid = make_grid()
    # a State carrying only a dummy scalar, none of the named ones
    only = State({"dummy": grid.create_field(
        fr.spatial.Collocated().resolve(grid), name="dummy")})
    with pytest.raises(MissingComponentError, match=match):
        getattr(only, accessor)


# ================================================================
#  Chart-native view (state.chart, ruling (d))
# ================================================================
def test_chart_is_identity_on_a_flat_grid():
    # off a mapped grid every chart component is the stored physical
    # field (the horizontal components are always the identity).
    state = make_model().state
    assert state.chart["w"] is state["w"]
    assert state.chart["u"] is state["u"]
    assert state.chart["v"] is state["v"]
    # attribute access mirrors item access
    assert state.chart.w is state["w"]
    assert state.chart.u is state["u"]


def test_chart_velocities_destructure_in_axis_order():
    state = make_model().state
    u, v, w = state.chart.velocities
    assert u is state["u"]
    assert v is state["v"]
    assert w is state["w"]


def test_chart_is_read_only():
    chart = make_model().state.chart
    with pytest.raises(TypeError, match="read-only"):
        chart["w"] = None
    with pytest.raises(AttributeError, match="read-only"):
        chart.w = None


# ================================================================
#  Parameter-free diagnostics: rel_vort_z and hor_divergence
# ================================================================
def test_diagnostics_are_zero_at_rest():
    model = make_model()  # a freshly assembled model starts at rest
    state = model.state
    assert float(np.abs(state.rel_vort_z.data).max()) == 0.0
    assert float(np.abs(state.hor_divergence.data).max()) == 0.0


def test_diagnostics_are_finite_and_nonzero_on_a_set_state():
    model = make_model()
    rng = np.random.default_rng(0)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    state = model.state
    rv = np.asarray(state.rel_vort_z.data)
    div = np.asarray(state.hor_divergence.data)
    assert np.isfinite(rv).all()
    assert np.isfinite(div).all()
    assert np.abs(rv).max() > 0.0
    assert np.abs(div).max() > 0.0


def test_diagnostic_metadata_names():
    model = make_model()
    rng = np.random.default_rng(1)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    state = model.state
    assert state.rel_vort_z.name == "rel_vort_z"
    assert state.rel_vort_z.metadata.units == "1/s"
    assert state.hor_divergence.name == "hor_divergence"
    assert state.hor_divergence.metadata.units == "1/s"
