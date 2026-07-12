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
    SphericalCoriolis,
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


# ================================================================
#  SphericalCoriolis: the chart-grid rotation (stage C2)
# ================================================================
OMEGA = 1.7
LAT_MAX = float(np.deg2rad(80.0))


def sphere_grid(nlon=2 * N, nlat=N):
    """Lat-lon sphere chart (unit radius, polar caps excluded)."""
    mlon = fr.spatial.meshes.IntervalMesh(
        nlon, (0.0, float(2.0 * np.pi)), name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(
        nlat, (-LAT_MAX, LAT_MAX), periodic=False, name="lat")
    mapping = fr.spatial.CoordinateMapping(chart={
        "X": lambda lon, lat: (
            jnp.cos(lat) * jnp.cos(lon),
            jnp.cos(lat) * jnp.sin(lon),
            jnp.sin(lat))})
    grid = fr.spatial.Grid((mlon, mlat), mapping=mapping)
    grid.merge_overrides({
        "raise_index": fr.spatial.operators.RaiseIndex(
            ("lon", "lat"), diagonal=True),
        "lower_index": fr.spatial.operators.LowerIndex(
            ("lon", "lat"), diagonal=True)})
    return grid


def make_sphere(coriolis, csqr=0.7):
    """Build a spherical shallow-water model (linear terms only)."""
    return sw.Model(
        grid=sphere_grid(), coords=("lon", "lat"), csqr=csqr,
        rossby_number=0.2, coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=3))


def test_spherical_rejects_bad_coords():
    with pytest.raises(TypeError, match="two distinct strings"):
        SphericalCoriolis(coords=("lon",))
    with pytest.raises(TypeError, match="two distinct strings"):
        SphericalCoriolis(coords=("lat", "lat"))


def test_spherical_metric_weight_adds_a_field_reference():
    module = SphericalCoriolis(metric_weight="csqr")
    assert module.metric_weight == "csqr"
    assert tuple(ref.name for ref in module.field_references) == (
        "u", "v", "csqr")
    assert SphericalCoriolis().metric_weight is None


def test_spherical_declares_extra_halo_on_its_coords():
    module = SphericalCoriolis(coords=("a", "b"))
    assert module.coords == ("a", "b")
    assert module.extra_halo["a"] == 1
    assert module.extra_halo["b"] == 1


def test_spherical_requires_a_chart_grid():
    with pytest.raises(ValueError, match="chart-coupled grid"):
        make_channel(0.7, SphericalCoriolis(omega=OMEGA))


def test_spherical_coords_must_match_the_chart():
    with pytest.raises(ValueError, match="chart coordinates"):
        sw.Model(
            grid=sphere_grid(), coords=("lon", "lat"), csqr=0.7,
            coriolis=SphericalCoriolis(
                omega=OMEGA, coords=("phi", "theta")),
            advection=False,
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3))


def test_spherical_f_field_is_2_omega_sin_lat():
    model = make_sphere(SphericalCoriolis(omega=OMEGA))
    f = model.state["f_coriolis"]
    lat = model.grid.evaluation_nodes(f.function_space, "lat").data
    np.testing.assert_allclose(
        np.asarray(f.data), np.asarray(2.0 * OMEGA * jnp.sin(lat)),
        rtol=0.0, atol=1e-14)


def test_spherical_rotation_is_the_physical_rotation():
    # solid-body zonal flow u^lon = w0, v = 0: the term must return
    # du = 0 and dv^lat = -f sqrt(g) g^latlat u = -2 Omega w0
    # sin(lat) cos(lat) (unit sphere) up to interpolation error
    # (measured: 0.042 / 0.011 / 0.005 absolute at nlat 8/16/32
    # against a 0.5 scale)
    errors = []
    w0 = 0.3
    coriolis_term = fr.model.term_predicates.named(
        "SphericalCoriolis/coriolis")
    for nlat in (N, 2 * N):
        model = sw.Model(
            grid=sphere_grid(nlon=2 * N, nlat=nlat),
            coords=("lon", "lat"), csqr=0.7, rossby_number=0.2,
            coriolis=SphericalCoriolis(omega=OMEGA),
            advection=False,
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3))
        model.set_fields(
            u=lambda lon, lat: w0 + 0.0 * lon + 0.0 * lat)
        dz = model.tendency(model.state, filter=coriolis_term)
        assert float(np.abs(np.asarray(dz["u"].data)).max()) < 1e-14
        v = model.state["v"]
        lat = model.grid.evaluation_nodes(
            v.function_space, "lat").data
        exact = -2.0 * OMEGA * w0 * np.sin(lat) * np.cos(lat)
        errors.append(float(np.abs(
            np.asarray(dz["v"].data) - np.asarray(exact)).max()))
    assert errors[0] < 5e-2
    assert errors[1] < 0.4 * errors[0]  # refinement converges


@pytest.mark.parametrize("weight", [None, "csqr"])
def test_spherical_rotation_is_m_skew_on_the_sphere(weight):
    # rotation does no work under the metric energy
    # E = sum sqrt(g) g_ii w (u^i)^2 / 2 (measure-weighted sums):
    # the flux weight G transposes exactly across the .to averages
    model = make_sphere(SphericalCoriolis(
        omega=OMEGA, metric_weight=weight))
    rng = np.random.default_rng(4)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    coriolis_term = fr.model.term_predicates.named(
        "SphericalCoriolis/coriolis")
    z = model.state
    dz = model.tendency(z, filter=coriolis_term)
    grid = model.grid
    u, v = z["u"], z["v"]
    g_uu = grid.metric(u.function_space.bare, "g_lonlon")
    g_vv = grid.metric(v.function_space.bare, "g_latlat")
    w_u = z["csqr"].to(u) if weight else 1.0
    w_v = z["csqr"].to(v) if weight else 1.0
    rate_u = float((g_uu * w_u * u * dz["u"])
                   .integrate().data.ravel()[0])
    rate_v = float((g_vv * w_v * v * dz["v"])
                   .integrate().data.ravel()[0])
    assert abs(rate_u + rate_v) / (abs(rate_u) + abs(rate_v)) < 1e-14
