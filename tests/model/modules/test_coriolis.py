"""The shared Coriolis modules: references, term forms, skewness.

The metric_weight knob (the thickness-weighted rotation): with a
varying velocity energy weight w(y) — the variable-depth shallow
water csqr — the unweighted staggered rotation does work against the
diag(w, w, 1) metric (the u and v nodes sit at different y), while
the weighted flux form ``v -= (w_u f_u u).to(v) / w_v`` is exactly
M-skew for any f and any positive w profile (the linearized
Sadourny/Arakawa pairing). For a constant weight both forms coincide.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.energy import EnergyMetric
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)

N = 8
F0 = 1.3


def csqr_fn(y):
    return 1.0 + 0.5 * jnp.tanh(4.0 * (y - 0.5))


def make_channel(csqr, coriolis):
    """Build a linear walled channel with the given modules."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                     name="y")
    return sw.Model(
        grid=fr.spatial.Grid((mx, my)), csqr=csqr, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def random_state(model, seed):
    """Random data on the prognostic components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(
            np.asarray(model.state[name].data).shape)
        for name in ("u", "v", "p")})
    return sw.State({name: model.state[name]
                     for name in ("u", "v", "p")})


# ================================================================
#  Declarations: the metric-weight reference
# ================================================================
@pytest.mark.parametrize("cls", [FPlaneCoriolis, BetaPlaneCoriolis])
def test_metric_weight_defaults_to_none(cls):
    module = cls()
    assert module.metric_weight is None
    assert tuple(ref.name for ref in module.field_references) == (
        "u", "v")


@pytest.mark.parametrize("cls", [FPlaneCoriolis, BetaPlaneCoriolis])
def test_metric_weight_adds_a_field_reference(cls):
    module = cls(metric_weight="csqr")
    assert module.metric_weight == "csqr"
    assert tuple(ref.name for ref in module.field_references) == (
        "u", "v", "csqr")


# ================================================================
#  The term: constant weight coincides with the unweighted form
# ================================================================
def test_weighted_rotation_matches_unweighted_for_constant_csqr():
    plain = make_channel(0.7, FPlaneCoriolis(f0=F0))
    weighted = make_channel(0.7, FPlaneCoriolis(
        f0=F0, metric_weight="csqr"))
    rng = np.random.default_rng(1)
    fields = {name: rng.standard_normal(
        np.asarray(plain.state[name].data).shape)
        for name in ("u", "v", "p")}
    plain.set_fields(**fields)
    weighted.set_fields(**fields)
    t_plain = plain.tendency(plain.state)
    t_weighted = weighted.tendency(weighted.state)
    for c in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(t_weighted[c].data),
            np.asarray(t_plain[c].data), atol=1e-14)


# ================================================================
#  Skewness under the weighted metric (the reason the knob exists)
# ================================================================
def energy_rate(model, seed):
    """Return |Re<z, Lz>_M| / |z|_M^2 under the model's own metric."""
    z = random_state(model, seed)
    metric = EnergyMetric.from_model(
        model, require_constant_coriolis=False,
        allow_field_weights=True)
    tendency = model.tendency(model.state)
    t = sw.State({c: tendency[c] for c in ("u", "v", "p")})
    rate = float(np.real(complex(metric.inner(z, t))))
    return abs(rate) / float(np.real(complex(metric.inner(z, z))))


def test_weighted_rotation_conserves_the_weighted_energy():
    # varying csqr + metric_weight: Re<z, Lz>_M = 0 to machine
    # precision (rotation does no work; gravity pairs through the
    # div/grad transposes with c^2 inside the flux)
    model = make_channel(csqr_fn, FPlaneCoriolis(
        f0=F0, metric_weight="csqr"))
    assert energy_rate(model, seed=2) < 1e-14


def test_unweighted_rotation_does_work_against_a_varying_metric():
    # the documented failure mode the knob repairs: without the
    # thickness weighting the rotation energy leak is O(dy c^2').
    # Explicit assembly (the sw.Model preset refuses this pairing).
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                     name="y")
    model = fr.model.Model(
        grid=fr.spatial.Grid((mx, my)),
        modules=(
            sw.modules.DynamicalCore(csqr=csqr_fn,
                                     rossby_number=0.2),
            FPlaneCoriolis(f0=F0)),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    assert energy_rate(model, seed=2) > 1e-6


def test_beta_weighted_rotation_conserves_the_weighted_energy():
    model = make_channel(csqr_fn, BetaPlaneCoriolis(
        f0=F0, beta=2.0, metric_weight="csqr"))
    assert energy_rate(model, seed=3) < 1e-14
