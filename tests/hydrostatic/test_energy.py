"""The hydrostatic energy-weight builder (the model's diag(M))."""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=6, depth=1.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid, *, n2=2.0, gravity=1.5, f0=1.3):
    """Return a linear hydrostatic model with the explicit free surface."""
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=gravity),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


def random_state(model, seed=3):
    """Fill every prognostic component with standard-normal noise."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(model.state[name].shape)
        for name in ("u", "v", "b", "ps")})
    return model.state


def relative_skew(metric, model, state):
    """Return |<X, M dX/dt>| over the size of its per-component terms."""
    rate = model.tendency(state)
    terms = [fr.model.EnergyMetric({name: weight}).inner(state, rate).real
             for name, weight in metric.weights.items()]
    return abs(sum(terms)) / sum(abs(term) for term in terms)


# ================================================================
#  The weight map
# ================================================================
def test_weights_are_diag_1_1_inv_n2_ps_weight():
    weights = hy.energy.hydrostatic_energy_weights(0.5, 0.25)
    assert set(weights) == {"u", "v", "b", "ps"}
    assert weights["u"] == 1.0
    assert weights["v"] == 1.0
    assert weights["b"] == 0.5
    assert weights["ps"] == 0.25


def test_weights_accept_a_profile_field_for_inv_n2():
    grid = make_grid(nx=4, nz=4)
    inv_n2 = grid.create_field(
        fr.spatial.Collocated().resolve(grid),
        init=lambda x, y, z: 1.0 + 0.0 * (x + y + z))
    weights = hy.energy.hydrostatic_energy_weights(inv_n2, 0.25)
    assert weights["b"] is inv_n2
    assert np.asarray(weights["b"].data).shape == \
        np.asarray(inv_n2.data).shape
    assert weights["ps"] == 0.25


# ================================================================
#  The ps weight is depth-integrated: H/c^2 = 1/g, not 1/c^2
# ================================================================
def test_from_model_ps_weight_is_the_depth_integrated_reciprocal():
    depth, gravity = 2.0, 1.5
    model = make_model(make_grid(depth=depth), gravity=gravity)
    metric = fr.model.EnergyMetric.from_model(model)
    csqr = gravity * depth
    assert metric.weights["ps"] == pytest.approx(depth / csqr)
    assert metric.weights["ps"] == pytest.approx(1.0 / gravity)
    assert metric.weights["ps"] != pytest.approx(1.0 / csqr)


def test_depth_integrated_weights_keep_the_linear_operator_skew():
    # the H2 energy gate through the Profile quadrature of ps (no
    # broadcast onto the 3D volume): <X, M dX/dt> vanishes to round-off
    # only with the depth in the ps weight
    depth, n2, gravity = 2.0, 2.0, 1.5
    model = make_model(make_grid(depth=depth), n2=n2, gravity=gravity)
    state = random_state(model)

    by_hand = fr.model.EnergyMetric(
        hy.energy.hydrostatic_energy_weights(1.0 / n2, 1.0 / gravity))
    assert relative_skew(by_hand, model, state) < 1e-12
    assert relative_skew(
        fr.model.EnergyMetric.from_model(model), model, state) < 1e-12

    # the bare 1/c^2 weight misses the column depth and is not an energy
    csqr = gravity * depth
    wrong = fr.model.EnergyMetric(
        hy.energy.hydrostatic_energy_weights(1.0 / n2, 1.0 / csqr))
    assert relative_skew(wrong, model, state) > 1e-3
