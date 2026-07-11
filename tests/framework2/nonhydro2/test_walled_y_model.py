"""The walled-y nonhydro channel model (x, z periodic, y bounded).

The rotating stratified channel is the first configuration combining
lateral walls with the nonhydro pressure CONSTRAINT stage. The tests
pin the topology-driven assembly (the wall-normal ``v`` derives the
Dirichlet tag on the inner y faces, everything else stays BC-free),
the walled-y spectral pressure solve (the projection drives the
discrete divergence to machine zero with no boundary seams), the
projected-tendency matvec, and time stepping. A nonlinear run
(``advection=True``: the centered scheme's structural-zero wall
fluxes) and a walled-x twin guarding the axis-generic gradient retag
in the projection stage ride on top.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.bc import BC
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.spaces.nodal import NodeSet

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0
COMPONENTS = ("u", "v", "w", "b")


def make_model(*, walled="y"):
    """Build a linear nonhydro channel with one bounded axis."""
    meshes = tuple(
        IntervalMesh(N, (0.0, 1.0 if name == walled else 2 * np.pi),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    return nh.Model(
        grid=Grid(meshes), advection=False, dsqr=DSQR,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def model():
    """One walled-y channel model shared across the module."""
    return make_model()


def _random_state(model, seed):
    """Write random prognostic fields onto the model."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(model.state[c].data.shape)
        for c in COMPONENTS})
    return model.state


def _max_divergence(state):
    """Max |div(u, v, w)| over the whole domain (walls included)."""
    div = Divergence()(VectorField(
        {c: state[c] for c in ("u", "v", "w")}))
    return float(np.abs(np.asarray(div.data)).max())


# ================================================================
#  Topology-driven assembly (grid periodicity is the only switch)
# ================================================================
def test_wall_normal_velocity_derives_the_dirichlet_tag(model):
    # v lives on the INNER y faces with the derived Dirichlet wall
    # tag; u, w, b keep the full y-center column, BC-free
    v_factor = model.state["v"].function_space.factor("y")
    assert v_factor.node_set is NodeSet.INNER
    assert all(c is BC.DIRICHLET for c in v_factor.bc.components)
    assert model.state["v"].data.shape == (N, N - 1, N)
    for c in ("u", "w", "b"):
        assert model.state[c].data.shape == (N, N, N)
        assert all(bc is BC.NONE for bc in
                   model.state[c].function_space.factor("y")
                   .bc.components)


# ================================================================
#  The walled-y pressure projection (machine-zero divergence)
# ================================================================
def test_projection_drives_the_divergence_to_machine_zero(model):
    # the spectral solve on the Neumann-tagged y sibling: after the
    # CONSTRAINT stage the discrete divergence is machine zero on
    # EVERY cell — no boundary seams at the walls
    _random_state(model, seed=1)
    projected = model.constrain(model.state)
    assert _max_divergence(projected) < 1e-13


def test_projection_is_idempotent(model):
    _random_state(model, seed=2)
    once = model.constrain(model.state)
    twice = model.constrain(once)
    err = max(float(np.abs(np.asarray(twice[c].data)
                           - np.asarray(once[c].data)).max())
              for c in COMPONENTS)
    assert err < 1e-13


def test_constrained_tendency_is_divergence_free(model):
    # the H1 surface on the walled channel: the projected prognostic
    # tendency (no diagnostic 'p' in the result)
    _random_state(model, seed=3)
    tau = model.tendency(model.state, constraints=True)
    assert tau.component_names == COMPONENTS
    assert _max_divergence(tau) < 1e-13


# ================================================================
#  Time stepping
# ================================================================
def test_advance_keeps_the_state_divergence_free(model):
    _random_state(model, seed=4)
    model.advance(3)
    assert not model.panicked
    state = model.state
    assert all(np.isfinite(np.asarray(state[c].data)).all()
               for c in COMPONENTS)
    assert _max_divergence(state) < 1e-13


# ================================================================
#  The nonlinear channel (CenteredAdvection on walled grids)
# ================================================================
@pytest.mark.parametrize("walled", [
    pytest.param(("y",), id="channel-y"),
    pytest.param(("y", "z"), id="channel-and-lid"),
])
def test_nonlinear_advance_stays_finite_and_divergence_free(walled):
    # the walled model assembles WITH advection (the centered
    # scheme's wall fluxes are structural zeros) and a short
    # nonlinear run stays finite and divergence-clean
    meshes = tuple(
        IntervalMesh(N, (0.0, 1.0 if name in walled else 2 * np.pi),
                     periodic=(name not in walled), name=name)
        for name in ("x", "y", "z"))
    model = nh.Model(
        grid=Grid(meshes), advection=True, dsqr=DSQR,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    _random_state(model, seed=6)
    model.advance(3)
    assert not model.panicked
    state = model.state
    assert all(np.isfinite(np.asarray(state[c].data)).all()
               for c in COMPONENTS)
    assert _max_divergence(state) < 1e-13


# ================================================================
#  The walled-x twin (the axis-generic gradient retag)
# ================================================================
def test_walled_x_channel_projects_divergence_free():
    # walls on x instead: u derives the Dirichlet tag and the
    # projection stage retags grad p onto it (the axis-generic seam)
    model = make_model(walled="x")
    u_factor = model.state["u"].function_space.factor("x")
    assert u_factor.node_set is NodeSet.INNER
    assert all(c is BC.DIRICHLET for c in u_factor.bc.components)
    _random_state(model, seed=5)
    projected = model.constrain(model.state)
    assert _max_divergence(projected) < 1e-13
