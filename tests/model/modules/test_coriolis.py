"""The shared Coriolis modules: references, term forms, skewness.

RotationCoriolis derives the Coriolis parameter from the chart itself,
f = 2 Omega . n_hat with n_hat the surface unit normal: the lat-lon
sphere with Omega = (0, 0, w) reproduces the classical 2 w sin(lat),
the identity chart X = (x, y, 0) reproduces the f-plane bitwise, and a
torus gives the f of its analytic normal.

Rotation is opt-in: the presets' coriolis=None (the argument omitted)
installs no Coriolis module at all — no f_coriolis field, no rotation
term, no coriolis.f0 provide.

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
    RotationCoriolis,
    chart_rotation,
    linear_rotation,
)
from fridom.model.params import CORIOLIS_F0

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
#  The lat-lon sphere chart (stage C2)
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


# ================================================================
#  RotationCoriolis: f = 2 Omega . n_hat, derived from the chart
# ================================================================
#: torus geometry (major/minor radius) of the third sanity check
MAJOR, MINOR = 2.0, 0.5
#: a deliberately tilted rotation vector: f varies along BOTH torus
#: coordinates (a polar Omega would make it a function of "b" only)
TILTED = (0.3, -0.2, 1.1)


def chart_grid(m_1, m_2, chart):
    """Build a chart grid with the diagonal index moves."""
    grid = fr.spatial.Grid(
        (m_1, m_2), mapping=fr.spatial.CoordinateMapping(
            chart={"X": chart}))
    names = (m_1.names[0], m_2.names[0])
    grid.merge_overrides({
        "raise_index": fr.spatial.operators.RaiseIndex(
            names, diagonal=True),
        "lower_index": fr.spatial.operators.LowerIndex(
            names, diagonal=True)})
    return grid


def torus_grid(na=2 * N, nb=N):
    """Torus chart: both coordinates periodic, diagonal metric."""
    two_pi = float(2.0 * np.pi)
    return chart_grid(
        fr.spatial.meshes.IntervalMesh(na, (0.0, two_pi), name="a"),
        fr.spatial.meshes.IntervalMesh(nb, (0.0, two_pi), name="b"),
        lambda a, b: ((MAJOR + MINOR * jnp.cos(b)) * jnp.cos(a),
                      (MAJOR + MINOR * jnp.cos(b)) * jnp.sin(a),
                      MINOR * jnp.sin(b)))


def identity_chart_grid():
    """Build the flat identity chart X = (x, y, 0) in R^3."""
    return chart_grid(
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x"),
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                       name="y"),
        lambda x, y: (x, y, 0.0 * x))


def make_chart_model(grid, coriolis=None, *, coords, csqr=0.7):
    """Build a linear shallow-water model on a chart grid."""
    return sw.Model(
        grid=grid, coords=coords, csqr=csqr, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))


def rotation_energy_rate(model, module, coords, weight):
    """|dE/dt| / |E-scale| of the rotation term alone.

    The metric energy E = sum sqrt(g) g_ii w (u^i)^2 / 2 (with the
    measure-weighted sums of ``integrate``): the flux weight G
    transposes exactly across the ``.to`` averages, so the rate is
    machine zero for any f and any positive weight.
    """
    rng = np.random.default_rng(4)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    z = model.state
    dz = model.tendency(z, filter=fr.model.term_predicates.named(
        f"{module}/coriolis"))
    grid = model.grid
    c_1, c_2 = coords
    u, v = z["u"], z["v"]
    g_uu = grid.metric(u.function_space.bare, f"g_{c_1}{c_1}")
    g_vv = grid.metric(v.function_space.bare, f"g_{c_2}{c_2}")
    w_u = z["csqr"].to(u) if weight else 1.0
    w_v = z["csqr"].to(v) if weight else 1.0
    rate_u = float((g_uu * w_u * u * dz["u"])
                   .integrate().data.ravel()[0])
    rate_v = float((g_vv * w_v * v * dz["v"])
                   .integrate().data.ravel()[0])
    return abs(rate_u + rate_v) / (abs(rate_u) + abs(rate_v))


# ----------------------------------------------------------------
#  Construction-time validation
# ----------------------------------------------------------------
def test_rotation_omega_is_the_ambient_vector():
    with pytest.raises(TypeError, match="AMBIENT rotation vector"):
        RotationCoriolis(omega=1.0)
    with pytest.raises(TypeError, match="AMBIENT rotation vector"):
        RotationCoriolis(omega=(0.0, 1.0))
    module = RotationCoriolis(omega=(0.0, 0.0, OMEGA))
    assert tuple(np.asarray(module.omega)) == (0.0, 0.0, OMEGA)
    assert module.coords == ("lon", "lat")
    assert module.metric_weight is None


def test_rotation_rejects_bad_coords():
    with pytest.raises(TypeError, match="two distinct strings"):
        RotationCoriolis(coords=("lon",))
    with pytest.raises(TypeError, match="two distinct strings"):
        RotationCoriolis(coords=("lat", "lat"))


def test_rotation_declares_its_references_and_halo():
    module = RotationCoriolis(coords=("a", "b"),
                              metric_weight="csqr")
    assert tuple(ref.name for ref in module.field_references) == (
        "u", "v", "csqr")
    assert module.extra_halo["a"] == 1
    assert module.extra_halo["b"] == 1


def test_rotation_requires_a_chart_grid():
    with pytest.raises(ValueError, match="chart-coupled grid"):
        make_channel(0.7, RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                                           coords=("x", "y")))


def test_rotation_coords_must_match_the_chart():
    with pytest.raises(ValueError, match="chart coordinates"):
        make_chart_model(
            sphere_grid(),
            RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                             coords=("phi", "theta")),
            coords=("lon", "lat"))


def test_rotation_rejects_a_nonorthogonal_chart():
    # a sheared chart: g_xy = 0.4 != 0. The off-diagonal rotation has
    # no energy-conserving staggered form, so the module refuses.
    grid = chart_grid(
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x"),
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                       name="y"),
        lambda x, y: (x + 0.4 * y, y, 0.0 * x))
    with pytest.raises(ValueError, match="ORTHOGONAL chart"):
        make_chart_model(
            grid, RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                                   coords=("x", "y")),
            coords=("x", "y"))


# ----------------------------------------------------------------
#  Sanity check 1: the lat-lon sphere reproduces 2 Omega sin(lat)
# ----------------------------------------------------------------
def test_rotation_on_the_sphere_is_2_omega_sin_lat():
    model = make_chart_model(
        sphere_grid(), RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                                        coords=("lon", "lat")),
        coords=("lon", "lat"))
    f = model.state["f_coriolis"]
    lat = model.grid.evaluation_nodes(f.function_space, "lat").data
    np.testing.assert_allclose(
        np.asarray(f.data),
        np.asarray(2.0 * OMEGA * jnp.sin(lat) + 0.0 * f.data),
        rtol=0.0, atol=1e-14)


def test_rotation_on_the_sphere_is_the_physical_rotation():
    # the spherical physics, pinned on the chart-generic module (the
    # only rotation module there is): solid-body zonal flow
    # u^lon = w0, v = 0 must give du = 0 and
    # dv^lat = -f sqrt(g) g^latlat u = -2 Omega w0 sin(lat) cos(lat)
    # (unit sphere) up to interpolation error (measured: 0.042 /
    # 0.011 / 0.005 absolute at nlat 8/16/32 against a 0.5 scale)
    errors = []
    w0 = 0.3
    coriolis_term = fr.model.term_predicates.named(
        "RotationCoriolis/coriolis")
    for nlat in (N, 2 * N):
        model = make_chart_model(
            sphere_grid(nlon=2 * N, nlat=nlat),
            RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                             coords=("lon", "lat")),
            coords=("lon", "lat"))
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


# ----------------------------------------------------------------
#  Sanity check 2: the identity chart IS the f-plane (bitwise)
# ----------------------------------------------------------------
def test_rotation_on_the_identity_chart_is_the_f_plane():
    # X = (x, y, 0) -> n_hat = (0, 0, 1) exactly, so f = 2 Omega and
    # every metric factor is an exact 1.0: the chart-generic module
    # reproduces FPlaneCoriolis(f0=2 Omega) BIT FOR BIT
    omega_z = 0.65
    chart = make_chart_model(
        identity_chart_grid(),
        RotationCoriolis(omega=(0.0, 0.0, omega_z),
                         coords=("x", "y")),
        coords=("x", "y"))
    assert (np.asarray(chart.state["f_coriolis"].data)
            == 2.0 * omega_z).all()
    flat = make_channel(0.7, FPlaneCoriolis(f0=2.0 * omega_z))
    rng = np.random.default_rng(12)
    fields = {name: rng.standard_normal(
        np.asarray(chart.state[name].data).shape)
        for name in ("u", "v", "p")}
    chart.set_fields(**fields)
    flat.set_fields(**fields)
    dc = chart.tendency(chart.state, filter=(
        fr.model.term_predicates.named("RotationCoriolis/coriolis")))
    df = flat.tendency(flat.state, filter=(
        fr.model.term_predicates.named("FPlaneCoriolis/coriolis")))
    for comp in ("u", "v"):
        assert np.array_equal(np.asarray(dc[comp].data),
                              np.asarray(df[comp].data))


# ----------------------------------------------------------------
#  Sanity check 3: the torus f is 2 Omega . n_hat of its normal
# ----------------------------------------------------------------
def test_rotation_on_the_torus_matches_the_analytic_normal():
    # X_a x X_b = r (R + r cos b) (cos b cos a, cos b sin a, sin b),
    # so n_hat is the bracket and f = 2 Omega . n_hat varies along
    # BOTH coordinates for the tilted Omega
    model = make_chart_model(
        torus_grid(), RotationCoriolis(omega=TILTED,
                                       coords=("a", "b")),
        coords=("a", "b"))
    f = model.state["f_coriolis"]
    a = model.grid.evaluation_nodes(f.function_space, "a").data
    b = model.grid.evaluation_nodes(f.function_space, "b").data
    exact = 2.0 * (TILTED[0] * jnp.cos(b) * jnp.cos(a)
                   + TILTED[1] * jnp.cos(b) * jnp.sin(a)
                   + TILTED[2] * jnp.sin(b))
    np.testing.assert_allclose(
        np.asarray(f.data), np.asarray(exact), rtol=0.0, atol=1e-14)
    # a genuinely two-dimensional f field (not a meridional profile)
    assert np.asarray(f.data).shape == (2 * N, N)


# ----------------------------------------------------------------
#  The load-bearing property: exact discrete skew-symmetry
# ----------------------------------------------------------------
@pytest.mark.parametrize("weight", [None, "csqr"])
def test_rotation_is_m_skew_on_the_sphere(weight):
    model = make_chart_model(
        sphere_grid(), RotationCoriolis(omega=(0.0, 0.0, OMEGA),
                                        coords=("lon", "lat"),
                                        metric_weight=weight),
        coords=("lon", "lat"))
    assert rotation_energy_rate(
        model, "RotationCoriolis", ("lon", "lat"), weight) < 1e-14


@pytest.mark.parametrize("weight", [None, "csqr"])
def test_rotation_is_m_skew_on_the_torus(weight):
    # the same machine-zero energy production with a tilted Omega,
    # i.e. an f varying along both chart coordinates
    model = make_chart_model(
        torus_grid(), RotationCoriolis(omega=TILTED, coords=("a", "b"),
                                       metric_weight=weight),
        coords=("a", "b"))
    assert rotation_energy_rate(
        model, "RotationCoriolis", ("a", "b"), weight) < 1e-14


# ================================================================
#  No rotation is the DEFAULT; the metric-blindness guards
# ================================================================
def test_omitting_coriolis_installs_no_rotation():
    # coriolis=None (the argument omitted) means NO Coriolis force:
    # no module, hence no f_coriolis field, no rotation term and no
    # coriolis.f0 provide. A rotating run names its rotation.
    model = make_channel(0.7, None)
    assert "f_coriolis" not in model.state
    assert CORIOLIS_F0 not in model.parameters
    assert not any(
        isinstance(m, FPlaneCoriolis | BetaPlaneCoriolis
                   | RotationCoriolis)
        for m in model._carry.modules)
    rng = np.random.default_rng(13)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    # no rotation term at all: the u tendency is pure gravity (zero
    # for a zero pressure field), and the run stays finite
    dz = model.tendency(model.state)
    assert float(np.abs(np.asarray(dz["u"].data)).max()) == 0.0
    model.advance(5)
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


def test_the_default_needs_no_grid_guard_on_a_chart_grid():
    # the default installs nothing, so it is metric-safe everywhere:
    # a chart grid without an explicit rotation is simply unrotating
    # (it used to raise, when the default was the f0 = 1 f-plane)
    model = make_chart_model(sphere_grid(), coriolis=None,
                             coords=("lon", "lat"))
    assert "f_coriolis" not in model.state
    assert CORIOLIS_F0 not in model.parameters


@pytest.mark.parametrize("cls", [FPlaneCoriolis, BetaPlaneCoriolis])
def test_metric_blind_coriolis_is_rejected_on_a_chart_grid(cls):
    with pytest.raises(ValueError, match="metric-blind"):
        make_chart_model(sphere_grid(), cls(),
                         coords=("lon", "lat"))


# ================================================================
#  The rotation expressions as shared functions (one owner)
# ================================================================
# The modules delegate their term bodies to `linear_rotation` /
# `chart_rotation`; the shallow-water energy correction subtracts
# exactly those functions, so the two can never drift apart.
@pytest.mark.parametrize("weight", [None, "csqr"])
def test_linear_rotation_is_the_module_term(weight):
    # the module's term body IS the shared function (the shallow-water
    # correction subtracts that function, so the two cannot drift)
    module = FPlaneCoriolis(f0=F0, metric_weight=weight)
    csqr = csqr_fn if weight is not None else 0.7
    model = make_channel(csqr, module)
    random_state(model, seed=3)
    state = model.state
    got = linear_rotation(state, metric_weight=weight)
    term = model.module(FPlaneCoriolis).coriolis(state, ctx=None)
    for name in ("u", "v"):
        assert np.array_equal(np.asarray(got[name].data),
                              np.asarray(term[name].data))


@pytest.mark.parametrize("weight", [None, "csqr"])
def test_chart_rotation_is_the_module_term(weight):
    coords = ("lon", "lat")
    module = RotationCoriolis(omega=(0.0, 0.0, 1.5), coords=coords,
                              metric_weight=weight)
    model = make_chart_model(sphere_grid(), module, coords=coords)
    random_state(model, seed=4)
    state = model.state
    got = chart_rotation(state, coords=coords, metric_weight=weight)
    term = model.module(RotationCoriolis).coriolis(state, ctx=None)
    for name in ("u", "v"):
        assert np.array_equal(np.asarray(got[name].data),
                              np.asarray(term[name].data))
