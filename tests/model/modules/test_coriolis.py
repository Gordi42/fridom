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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.model.energy import EnergyMetric
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
    RotationCoriolis,
    chart_rotation,
    linear_rotation,
)
from fridom.model.params import CORIOLIS_BETA, CORIOLIS_F0

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
    return fr.spatial.spherical.Grid(
        (nlon, nlat), lat_extent=(-LAT_MAX, LAT_MAX))


# ================================================================
#  RotationCoriolis: f = 2 Omega . n_hat, derived from the chart
# ================================================================
#: torus geometry (major/minor radius) of the third sanity check
MAJOR, MINOR = 2.0, 0.5
#: a deliberately tilted rotation vector: f varies along BOTH torus
#: coordinates (a polar Omega would make it a function of "b" only)
TILTED = (0.3, -0.2, 1.1)


def chart_grid(m_1, m_2, chart):
    """Build an orthogonal-chart grid (diagonal index moves)."""
    return fr.spatial.Grid(
        (m_1, m_2), mapping=fr.spatial.CoordinateMapping(
            chart={"X": chart}, orthogonal=True))


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
#  Sanity check 2: the identity chart IS the f-plane
# ----------------------------------------------------------------
def test_rotation_on_the_identity_chart_is_the_f_plane():
    # X = (x, y, 0) -> n_hat = (0, 0, 1) exactly, so f = 2 Omega and
    # every metric factor is an exact 1.0: the chart-generic module
    # reproduces FPlaneCoriolis(f0=2 Omega) -- to rounding.
    #
    # NOT bitwise, and deliberately so. Since the extra-halo gate
    # (`fix/sw-extra-halo-gate`, lever 8) `DynamicalCore.extra_halo`
    # is chart-conditional: the chart core requests 2 halo cells per
    # axis where the flat core requests none. The two models
    # therefore pad their storage differently, XLA fuses the `.to`
    # interpolation differently (FMA contraction in one path,
    # multiply-then-add in the other), and `v` lands ~16 ulp apart
    # (3e-15 relative). That is re-association, not physics:
    #  - the interior INPUTS are bitwise equal, only the outputs move;
    #  - the differences scatter across every interior column, not
    #    the wall-adjacent ones a bad ghost fill would touch;
    #  - equalising the two cores' `extra_halo` restores exact
    #    bitwise equality (0 ulp on both components).
    # `.data` is already halo-stripped, so narrowing to the interior
    # does not recover the bitwise claim -- the tolerance is the
    # honest assertion. It stays tight enough that any real error
    # (which would be O(dx) or worse) still fails it.
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
        got = np.asarray(dc[comp].data)
        want = np.asarray(df[comp].data)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-15)


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


# ================================================================
#  R1: time-dependent scalar f0 on the f-plane
# ================================================================
# The f-plane f is constant in space, so a Ramp-valued f0 is a
# SCALAR parameter (R1), not a field blend (R2): the AUXILIARY field
# is materialized at t=0 for a stable treedef, and the rotation term
# reads f0(t) from ctx.params (resolved at the stage clock, the same
# seam scaling.rossby rides) each step. The static (plain-float) path
# is untouched -- see test_linear_rotation_is_the_module_term above.
RAMP_DT = 5e-3


def _r1_grid():
    """Return a small periodic-x / walled-y channel (R1 fixture grid)."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                        name="y")
    return fr.spatial.Grid((mx, my))


def _r1_channel(grid, f0, order=3):
    """Return a linear sw channel with the given (float or Ramp) f0."""
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            RAMP_DT, order=order))


def test_fplane_ramped_f0_materializes_the_field_at_t0():
    """A Ramp f0 fills f_coriolis with f0(0): no jnp.full TypeError."""
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.6, 1.4, period=1.0, curve="cosine")
    model = _r1_channel(grid, ramp, order=1)
    field = np.asarray(model.state["f_coriolis"].data)
    np.testing.assert_allclose(field, float(ramp.at_time(0.0)))


@pytest.mark.parametrize("t", [0.0, 0.017, 0.05, 0.2])
def test_fplane_ramped_f0_tendency_equals_constant_model_at_stage_time(t):
    """ramped.tendency(z, t) == const-f0-model.tendency(z), bitwise.

    This is the 'per-step constant f resolved from the Ramp' claim:
    the term reads f0(t) at the stage clock, so its tendency equals a
    plain-float model built with f0 = ramp(t). The constant-in-space
    override and the field path agree bit-for-bit for a constant f.
    """
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.6, 1.4, period=0.05, curve="exp")
    ramped = _r1_channel(grid, ramp, order=1)
    z = random_state(ramped, seed=5)
    got = ramped.tendency(z, t=t)

    const = _r1_channel(grid, float(ramp.at_time(t)), order=1)
    const.set_fields(**{c: np.asarray(z[c].data) for c in ("u", "v", "p")})
    z_const = sw.State({c: const.state[c] for c in ("u", "v", "p")})
    want = const.tendency(z_const)
    for c in ("u", "v", "p"):
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_fplane_ramped_f0_matches_a_hand_stepped_ab3_oracle():
    """f0=Ramp advances under AB3 and matches a hand-stepped oracle.

    'Correct stage time' for AdamBashforth: the tendency is evaluated
    at the PRE-TICK clock (adam_bashforth.py step()), so step n (from
    a reset clock of 0) reads f0 at t_n = n*dt. The oracle rebuilds,
    per step, a CONSTANT-f0 model with f0 = ramp(t_n) and drives the
    identical AB3 warm-up combination by hand -- the AB weights do not
    depend on f0, so the whole discrepancy is the per-step f the term
    reads.
    """
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.6, 1.4, period=8 * RAMP_DT, curve="exp")
    order, steps = 3, 4
    comps = ("u", "v", "p")

    ramped = _r1_channel(grid, ramp, order=order)
    state0 = random_state(ramped, seed=11)
    init = {c: np.asarray(state0[c].data).copy() for c in comps}
    ramped.reset()
    ramped.set_state(state0)
    ramped.advance(steps)
    got = {c: np.asarray(ramped.state[c].data) for c in comps}

    # AB warm-up rows for order 3 (_warmup_table, eps=None): row r is
    # the AB(r+1) coefficient row zero-padded; warmup = min(n, 2).
    table = ((1.0, 0.0, 0.0), (1.5, -0.5, 0.0), (23 / 12, -4 / 3, 5 / 12))
    ring = [{c: np.zeros_like(init[c]) for c in comps}
            for _ in range(order - 1)]
    state = {c: init[c].copy() for c in comps}
    for n in range(steps):
        const = _r1_channel(grid, float(ramp.at_time(n * RAMP_DT)),
                            order=order)
        const.set_fields(**{c: state[c] for c in comps})
        z = sw.State({c: const.state[c] for c in comps})
        tend = const.tendency(z)
        f_n = {c: np.asarray(tend[c].data) for c in comps}
        levels = (f_n, *ring)
        weights = table[min(n, order - 1)]
        for c in comps:
            state[c] = state[c] + RAMP_DT * sum(
                weights[j] * levels[j][c] for j in range(order))
        ring = (f_n, *ring)[:order - 1]

    for c in comps:
        np.testing.assert_allclose(got[c], state[c], rtol=1e-11,
                                   atol=1e-13)


def test_fplane_ramped_f0_actually_changes_the_answer():
    """Sanity: the ramp matters -- a ramped run differs from frozen f0."""
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.3, 2.0, period=6 * RAMP_DT, curve="cosine")
    comps = ("u", "v", "p")

    ramped = _r1_channel(grid, ramp, order=3)
    state0 = random_state(ramped, seed=2)
    init = {c: np.asarray(state0[c].data).copy() for c in comps}
    ramped.reset()
    ramped.set_state(state0)
    ramped.advance(6)
    ramped_out = {c: np.asarray(ramped.state[c].data) for c in comps}

    frozen = _r1_channel(grid, float(ramp.at_time(0.0)), order=3)
    frozen.set_fields(**init)
    frozen.reset()
    frozen.set_state(sw.State({c: frozen.state[c] for c in comps}))
    frozen.advance(6)
    frozen_out = {c: np.asarray(frozen.state[c].data) for c in comps}

    assert any(not np.allclose(ramped_out[c], frozen_out[c])
               for c in comps)


def test_time_dependent_linear_parameters_reports_only_a_ramped_f0():
    """The AR-D7 hook: () for a float f0, (coriolis.f0,) for a Ramp."""
    assert FPlaneCoriolis(f0=F0).time_dependent_linear_parameters() == ()
    ramped = FPlaneCoriolis(f0=fr.model.Ramp(0.0, 1.0, period=1.0))
    assert ramped.time_dependent_linear_parameters() == (str(CORIOLIS_F0),)


def test_static_f0_never_touches_ctx_and_reports_nothing():
    """Gate (c): a plain-float f0 keeps the field path (ctx untouched)."""
    module = FPlaneCoriolis(f0=F0)
    # None override => the field path; the static branch dispatches on
    # the leaf type host-side and never dereferences ctx (ctx=None ok)
    assert module._stage_scalar_f(None) is None
    assert module.time_dependent_linear_parameters() == ()


def test_static_config_assembly_fingerprint_is_deterministic():
    """Gate (c): the float-f0 assembly fingerprint is stable."""
    digest1 = _r1_channel(_r1_grid(), F0, order=3).fingerprint.digest
    digest2 = _r1_channel(_r1_grid(), F0, order=3).fingerprint.digest
    assert digest1 == digest2


# ================================================================
#  R2: the beta-plane field blend f(y,t) = f0(t)*1 + beta(t)*y (AR-D2)
# ================================================================
# A ramped f0/beta makes the Coriolis parameter a spatial FIELD, so it
# is a FieldBlend of two assembly-materialized profiles (the constant
# unit and the meridional coordinate) weighted by f0(t)/beta(t) read at
# the stage clock. The static (plain-float) path is untouched.
def _beta_channel(grid, f0=F0, beta=0.0, order=3):
    """Return a linear sw channel with the given (float/Ramp) f0/beta."""
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=BetaPlaneCoriolis(f0=f0, beta=beta), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            RAMP_DT, order=order))


def test_beta_plane_static_declares_only_f_coriolis():
    """A plain-float beta-plane declares the single f_coriolis field."""
    model = _beta_channel(_r1_grid(), beta=2.0)
    assert "f_coriolis" in model.state
    assert "f_coriolis_const" not in model.state
    assert "f_coriolis_grad" not in model.state
    assert not model.module(BetaPlaneCoriolis)._blend_active


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"f0": fr.model.Ramp(0.6, 1.4, period=1.0)}, id="f0"),
     pytest.param({"beta": fr.model.Ramp(0.0, 2.0, period=1.0)},
                  id="beta")])
def test_beta_plane_ramped_declares_the_blend_ingredients(kwargs):
    """A ramped f0/beta declares the two FieldBlend ingredient fields."""
    model = _beta_channel(_r1_grid(), **kwargs)
    for name in ("f_coriolis", "f_coriolis_const", "f_coriolis_grad"):
        assert name in model.state
    assert model.module(BetaPlaneCoriolis)._blend_active


@pytest.mark.parametrize("t", [0.0, 0.017, 0.05, 0.2])
def test_beta_plane_ramped_tendency_equals_static_at_stage_time(t):
    """ramped.tendency(z, t) == static-beta(t) model, bitwise.

    The beta-plane counterpart of the f-plane R1 oracle: the rotation
    term reads the stage-time blend f(y,t) = f0 + beta(t)*y, so its
    tendency equals a plain-float model built with beta = ramp(t).
    """
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.0, 2.0, period=0.05, curve="exp")
    ramped = _beta_channel(grid, f0=1.3, beta=ramp, order=1)
    z = random_state(ramped, seed=5)
    got = ramped.tendency(z, t=t)

    const = _beta_channel(grid, f0=1.3, beta=float(ramp.at_time(t)),
                          order=1)
    const.set_fields(**{c: np.asarray(z[c].data)
                        for c in ("u", "v", "p")})
    z_const = sw.State({c: const.state[c] for c in ("u", "v", "p")})
    want = const.tendency(z_const)
    for c in ("u", "v", "p"):
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_beta_plane_ramped_f0_and_beta_together():
    """Both f0 and beta ramped: f(y,t) = f0(t) + beta(t)*y at stage.

    The oracle evaluates the ramp curves host-side (``at_time``) and
    the blend evaluates them traced, so a cosine curve can differ by a
    last-ulp (XLA vs eager transcendental rounding, not the blend); the
    tolerance is tight. The bit-exact claim is the single-ramp test.
    """
    grid = _r1_grid()
    r_f0 = fr.model.Ramp(0.6, 1.4, period=0.05, curve="cosine")
    r_beta = fr.model.Ramp(0.0, 2.0, period=0.05, curve="exp")
    ramped = _beta_channel(grid, f0=r_f0, beta=r_beta, order=1)
    z = random_state(ramped, seed=6)
    for t in (0.0, 0.03, 0.05):
        got = ramped.tendency(z, t=t)
        const = _beta_channel(grid, f0=float(r_f0.at_time(t)),
                              beta=float(r_beta.at_time(t)), order=1)
        const.set_fields(**{c: np.asarray(z[c].data)
                            for c in ("u", "v", "p")})
        z_const = sw.State({c: const.state[c] for c in ("u", "v", "p")})
        want = const.tendency(z_const)
        for c in ("u", "v", "p"):
            np.testing.assert_allclose(
                np.asarray(got[c].data), np.asarray(want[c].data),
                rtol=1e-13, atol=1e-14)


def test_beta_plane_ramped_actually_changes_the_answer():
    """Sanity: a ramped beta run differs from the frozen-beta run."""
    grid = _r1_grid()
    ramp = fr.model.Ramp(0.0, 3.0, period=6 * RAMP_DT, curve="cosine")
    comps = ("u", "v", "p")
    ramped = _beta_channel(grid, beta=ramp, order=3)
    state0 = random_state(ramped, seed=2)
    init = {c: np.asarray(state0[c].data).copy() for c in comps}
    ramped.advance(6)
    ramped_out = {c: np.asarray(ramped.state[c].data) for c in comps}

    frozen = _beta_channel(grid, beta=float(ramp.at_time(0.0)), order=3)
    frozen.set_fields(**init)
    frozen.advance(6)
    assert any(not np.allclose(
        ramped_out[c], np.asarray(frozen.state[c].data)) for c in comps)


def test_beta_plane_time_dependent_linear_parameters():
    """The AR-D7 hook: reports ramped f0/beta feeding the linear term."""
    assert BetaPlaneCoriolis(
        f0=F0, beta=2.0).time_dependent_linear_parameters() == ()
    ramp = fr.model.Ramp(0.0, 1.0, period=1.0)
    assert BetaPlaneCoriolis(
        f0=ramp).time_dependent_linear_parameters() == (str(CORIOLIS_F0),)
    assert BetaPlaneCoriolis(
        beta=ramp).time_dependent_linear_parameters() == (
        str(CORIOLIS_BETA),)
    both = BetaPlaneCoriolis(f0=ramp, beta=ramp)
    assert both.time_dependent_linear_parameters() == (
        str(CORIOLIS_F0), str(CORIOLIS_BETA))


def test_beta_plane_etdrk4_refuses_a_ramped_beta():
    """AR-D7: a frozen-L (ETDRK4) stepper refuses a ramped beta."""
    grid = _r1_grid()
    static = sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=BetaPlaneCoriolis(f0=1.0, beta=2.0), advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(RAMP_DT))
    basis = sw.eigenbasis(static)
    ramp = fr.model.Ramp(0.0, 2.0, period=1.0)
    with pytest.raises(
            fr.model.errors.TimeDependentLinearOperatorError,
            match=r"coriolis\.beta"):
        sw.Model(
            grid=grid, csqr=1.0, rossby_number=0.2,
            coriolis=BetaPlaneCoriolis(f0=1.0, beta=ramp),
            advection=True,
            time_stepper=fr.model.time_steppers.ETDRK4(RAMP_DT, basis),
            term_filter=~terms.linear)


def test_beta_plane_static_config_fingerprint_is_stable():
    """The float-beta assembly fingerprint is deterministic."""
    d1 = _beta_channel(_r1_grid(), beta=2.0).fingerprint.digest
    d2 = _beta_channel(_r1_grid(), beta=2.0).fingerprint.digest
    assert d1 == d2


@pytest.mark.multi_device
def test_beta_plane_blend_is_device_count_invariant(forced_devices):
    """Gate (d): the pointwise f(y) blend is halo-neutral (forced-4)."""
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    ramp = fr.model.Ramp(0.0, 2.0, period=5e-2, curve="exp")

    def build(device_ids):
        mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                            periodic=True, name="x")
        my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                            periodic=False, name="y")
        return _beta_channel(
            fr.spatial.Grid((mx, my), device_ids=device_ids),
            f0=1.0, beta=ramp, order=3)

    rng = np.random.default_rng(4)
    fields = {"u": rng.standard_normal((N, N)),
              "v": rng.standard_normal((N, N - 1)),
              "p": rng.standard_normal((N, N))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = build(device_ids)
        model.set_fields(**fields)
        model.advance(5)
        results[tag] = {c: np.asarray(model.state[c].data)
                        for c in ("u", "v", "p")}
        if tag == "many":
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in ("u", "v", "p")) < 1e-11
