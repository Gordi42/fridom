"""Spherical shallow water: assembly, metric terms, flat neutrality.

The stage-C2 unit surface of the shallowwater2 chart rework: the
same modules assemble on the lat-lon sphere chart (C-grid staggering
with the Dirichlet polar caps), the tendency terms match
hand-computed metric expressions built from raw ``diff``/``to`` field
operations plus ``grid.metric`` (independent of the composed dispatch
kinds), the flat path is results-neutral, and the taught errors
teach. The run-level physics (mass/energy conservation, TC2, the
flat-limit identity chart) lives in
``tests/validation/test_spherical_shallowwater.py``.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

from .conftest import make_grid

N = 8
CSQR = 0.7
RO = 0.4
OMEGA = 1.5
LAT_MAX = float(np.deg2rad(80.0))
TWO_PI = float(2.0 * np.pi)

ADVECT = fr.model.term_predicates.named("SadournyAdvection/advect")
GRAVITY = fr.model.term_predicates.named("DynamicalCore/gravity")


def sphere_grid(nlon=2 * N, nlat=N, radius=1.0):
    """Lat-lon sphere chart grid, polar caps excluded (the recipe)."""
    mlon = fr.spatial.meshes.IntervalMesh(nlon, (0.0, TWO_PI),
                                          name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(
        nlat, (-LAT_MAX, LAT_MAX), periodic=False, name="lat")
    mapping = fr.spatial.CoordinateMapping(chart={
        "X": lambda lon, lat: (
            radius * jnp.cos(lat) * jnp.cos(lon),
            radius * jnp.cos(lat) * jnp.sin(lon),
            radius * jnp.sin(lat))})
    grid = fr.spatial.Grid((mlon, mlat), mapping=mapping)
    grid.merge_overrides({
        "raise_index": fr.spatial.operators.RaiseIndex(
            ("lon", "lat"), diagonal=True),
        "lower_index": fr.spatial.operators.LowerIndex(
            ("lon", "lat"), diagonal=True)})
    return grid


def sphere_model(grid=None, *, advection=True, omega=OMEGA,
                 csqr=CSQR, ro=RO):
    """Assemble the spherical shallow-water model (the preset)."""
    if grid is None:
        grid = sphere_grid()
    return sw.Model(
        grid=grid, coords=("lon", "lat"), csqr=csqr,
        rossby_number=ro,
        coriolis=sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, omega), coords=("lon", "lat"),
            metric_weight="csqr"),
        advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))


def set_random(model, seed=3, amp=1.0):
    """Random prognostics (walls structural: v has no cap DOFs)."""
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=amp * rng.standard_normal(model.state["u"].shape),
        v=amp * rng.standard_normal(model.state["v"].shape),
        p=0.3 * amp * rng.standard_normal(model.state["p"].shape))


# ================================================================
#  Assembly: C-grid staggering on the sphere chart
# ================================================================
def test_assembles_with_the_sphere_staggering():
    grid = sphere_grid()
    model = sphere_model(grid)
    mlon, mlat = grid.factors
    u_space = model.state["u"].function_space.bare
    v_space = model.state["v"].function_space.bare
    p_space = model.state["p"].function_space.bare
    assert u_space.factor("lon") is mlon.right
    assert u_space.factor("lat") is mlat.center
    # the wall-normal velocity carries interior faces only, wall
    # Dirichlet (impermeability structural at the polar caps)
    assert v_space.factor("lon") is mlon.center
    assert v_space.factor("lat").node_set.name == "INNER"
    assert not v_space.factor("lat").bc.is_free
    assert p_space.factor("lon") is mlon.center
    assert p_space.factor("lat") is mlat.center


def test_polar_cap_wall_has_no_normal_dof():
    model = sphere_model()
    # nlat centers but only nlat - 1 interior faces: the cap value
    # is a boundary condition, not a degree of freedom
    assert model.state["v"].shape == (2 * N, N - 1)
    assert model.state["u"].shape == (2 * N, N)


# ================================================================
#  Tendency spot values against hand-built metric expressions
#  (raw diff/to + grid.metric only — independent of the seeded
#  composed kinds)
# ================================================================
def test_gravity_term_matches_the_hand_built_metric_form():
    model = sphere_model(advection=False)
    set_random(model, seed=5)
    z = model.state
    dz = model.tendency(z, filter=GRAVITY)
    grid = model.grid
    u, v, p, c = z["u"], z["v"], z["p"], z["csqr"]
    inv_uu = grid.metric(u.function_space.bare, "inv_g_lonlon")
    inv_vv = grid.metric(v.function_space.bare, "inv_g_latlat")
    du = -(inv_uu * p.diff("lon").retag(u))
    dv = -(inv_vv * p.diff("lat").retag(v))
    sq_u = grid.metric(u.function_space.bare, "sqrt_g")
    sq_v = grid.metric(v.function_space.bare, "sqrt_g")
    sq_p = grid.metric(p.function_space.bare, "sqrt_g")
    dp = -((sq_u * (c.to(u) * u)).diff("lon")
           + (sq_v * (c.to(v) * v)).diff("lat")) / sq_p
    np.testing.assert_allclose(np.asarray(dz["u"].data),
                               np.asarray(du.data), atol=1e-13)
    np.testing.assert_allclose(np.asarray(dz["v"].data),
                               np.asarray(dv.data), atol=1e-13)
    np.testing.assert_allclose(np.asarray(dz["p"].data),
                               np.asarray(dp.data), atol=1e-13)


def test_advection_term_matches_the_hand_built_metric_form():
    model = sphere_model()
    set_random(model, seed=7)
    z = model.state
    dz = model.tendency(z, filter=ADVECT)
    grid = model.grid
    u, v, p, c = z["u"], z["v"], z["p"], z["csqr"]
    p_full = c.to(p) + RO * p
    sq_u = grid.metric(u.function_space.bare, "sqrt_g")
    sq_v = grid.metric(v.function_space.bare, "sqrt_g")
    sq_p = grid.metric(p.function_space.bare, "sqrt_g")
    g_uu = grid.metric(u.function_space.bare, "g_lonlon")
    g_vv = grid.metric(v.function_space.bare, "g_latlat")
    inv_uu = grid.metric(u.function_space.bare, "inv_g_lonlon")
    inv_vv = grid.metric(v.function_space.bare, "inv_g_latlat")

    # thickness: -(Ro / sqrt_g) d_i(sqrt_g u^i p)
    dp = RO * -((sq_u * (u * p.to(u))).diff("lon")
                + (sq_v * (v * p.to(v))).diff("lat")) / sq_p

    # vorticity at the Dirichlet corner; q = zeta / h
    corner = u.function_space.bare.replace(
        lat=v.function_space.bare.factor("lat"))
    sq_c = grid.metric(corner, "sqrt_g")
    zeta = (((g_vv * v).diff("lon")).retag(corner)
            - ((g_uu * u).diff("lat")).retag(corner)) / sq_c
    q = zeta / p_full.to(zeta)
    fu = (sq_u * (u * p_full.to(u))).to(zeta)
    fv = (sq_v * (v * p_full.to(v))).to(zeta)
    ekin = 0.5 * ((sq_u * g_uu * u * u).to(p)
                  + (sq_v * g_vv * v * v).to(p)) / sq_p
    du = RO * (inv_uu * ((fv * q).to(u)
                         - ekin.diff("lon").retag(u)))
    dv = RO * (inv_vv * (-(fu * q).to(v)
                         - ekin.diff("lat").retag(v)))
    np.testing.assert_allclose(np.asarray(dz["p"].data),
                               np.asarray(dp.data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(dz["u"].data),
                               np.asarray(du.data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(dz["v"].data),
                               np.asarray(dv.data), atol=1e-12)


def test_semi_discrete_energy_rate_is_machine_zero_on_the_sphere():
    # gravity + Sadourny advection conserve the metric-weighted
    # thickness energy E = sum sqrt_g h g_ii (u^i)^2 / 2
    # + sqrt_g p^2 / 2 exactly (semi-discrete; measured 2.6e-17) —
    # the sphere twin of the Cartesian walled-grid gate
    model = sphere_model(omega=0.0)
    set_random(model, seed=11)
    z = model.state
    dz = model.tendency(z)
    grid = model.grid
    u, v, p = z["u"], z["v"], z["p"]
    du, dv, dp = dz["u"], dz["v"], dz["p"]
    h = z["csqr"].to(p) + RO * p
    g_uu = grid.metric(u.function_space.bare, "g_lonlon")
    g_vv = grid.metric(v.function_space.bare, "g_latlat")

    def total(f):
        return float(f.integrate().data.ravel()[0])

    terms = (
        total(g_uu * u * du * h.to(u)),
        total(0.5 * RO * g_uu * (u * u) * dp.to(u)),
        total(g_vv * v * dv * h.to(v)),
        total(0.5 * RO * g_vv * (v * v) * dp.to(v)),
        total(p * dp),
    )
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-13


def test_mass_rate_is_machine_zero_on_the_sphere():
    # impermeability at the caps + the flux-form metric divergence:
    # the sqrt_g-weighted mass rate telescopes exactly
    model = sphere_model()
    set_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14


# ================================================================
#  Flat-path results-neutrality
# ================================================================
def test_explicit_cartesian_coords_are_bitwise_neutral():
    # coords=("x", "y") spelled out == the defaults, bit for bit
    grid = make_grid()
    default = sw.Model(
        grid=grid, csqr=CSQR, rossby_number=RO,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))
    explicit = sw.Model(
        grid=grid, csqr=CSQR, rossby_number=RO, coords=("x", "y"),
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))
    rng = np.random.default_rng(2)
    fields = {
        "u": rng.standard_normal((16, 16)),
        "v": rng.standard_normal((16, 16)),
        "p": 0.3 * rng.standard_normal((16, 16))}
    default.set_fields(**fields)
    explicit.set_fields(**fields)
    d1 = default.tendency(default.state)
    d2 = explicit.tendency(explicit.state)
    for name in ("u", "v", "p"):
        assert np.array_equal(np.asarray(d1[name].data),
                              np.asarray(d2[name].data))


def test_flat_state_diagnostics_are_unchanged():
    # the generalized State sugar reproduces the Cartesian forms
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=CSQR, rossby_number=RO,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))
    rng = np.random.default_rng(6)
    model.set_fields(u=rng.standard_normal((16, 16)),
                     v=rng.standard_normal((16, 16)))
    z = model.state
    u, v = z["u"], z["v"]
    corner = u.function_space.bare.replace(
        y=v.function_space.bare.factor("y"))
    zeta = (v.diff("x").retag(corner) - u.diff("y").retag(corner))
    assert np.array_equal(np.asarray(z.rel_vort.data),
                          np.asarray(zeta.data))
    assert np.array_equal(
        np.asarray(z.divergence.data),
        np.asarray((u.diff("x") + v.diff("y")).data))
    # flat physical velocities are the prognostics themselves
    assert z.u_physical is u
    assert z.v_physical is v
    # and the bound energy diagnostics keep the Cartesian formulas
    center = z["p"].function_space
    u_c, v_c = u.to(center), v.to(center)
    assert np.array_equal(
        np.asarray(model.diagnostics.ekin().data),
        np.asarray(0.5 * (u_c.data**2 + v_c.data**2)))
    assert np.array_equal(
        np.asarray(model.diagnostics.epot().data),
        np.asarray(0.5 * z["p"].data**2 / CSQR))


# ================================================================
#  State sugar and diagnostics on the sphere
# ================================================================
def test_physical_velocities_scale_by_the_metric_root():
    model = sphere_model(grid=sphere_grid(radius=2.0))
    set_random(model, seed=9)
    z = model.state
    grid = model.grid
    u, v = z["u"], z["v"]
    # u_east = sqrt(g_lonlon) u^lon = a cos(lat) u^lon
    lat = grid.evaluation_nodes(u.function_space, "lat").data
    expected = np.asarray(2.0 * jnp.cos(lat) * u.data)
    np.testing.assert_allclose(np.asarray(z.u_physical.data),
                               expected, atol=1e-13)
    # v_north = sqrt(g_latlat) v^lat = a v^lat
    np.testing.assert_allclose(np.asarray(z.v_physical.data),
                               np.asarray(2.0 * v.data), atol=1e-13)
    assert z.u_physical.metadata.units == "m/s"


def test_solid_body_rotation_diagnostics():
    # u^lon = w0: divergence exactly zero (sqrt_g lon-independent),
    # vorticity 2 w0 sin(lat) to truncation
    model = sphere_model()
    w0 = 0.4
    model.set_fields(u=lambda lon, lat: w0 + 0.0 * lon + 0.0 * lat)
    z = model.state
    assert float(np.abs(np.asarray(z.divergence.data)).max()) < 1e-13
    zeta = z.rel_vort
    lat = model.grid.evaluation_nodes(
        zeta.function_space, "lat").data
    exact = 2.0 * w0 * np.sin(np.asarray(lat))
    err = np.abs(np.asarray(zeta.data) - exact).max()
    assert err < 5e-2  # O(dlat^2) at nlat = 8


def test_ekin_diagnostic_uses_the_metric_on_the_sphere():
    model = sphere_model()
    set_random(model, seed=13)
    z = model.state
    grid = model.grid
    center = z["p"].function_space
    u_c = z["u"].to(center)
    v_c = z["v"].to(center)
    g_uu = grid.metric(center.bare, "g_lonlon")
    g_vv = grid.metric(center.bare, "g_latlat")
    expected = 0.5 * (g_uu * u_c * u_c + g_vv * v_c * v_c)
    got = model.diagnostics.ekin()
    np.testing.assert_allclose(np.asarray(got.data),
                               np.asarray(expected.data),
                               atol=1e-13)


# ================================================================
#  Taught errors
# ================================================================
def test_core_rejects_bad_coords():
    with pytest.raises(TypeError, match="two distinct strings"):
        sw.modules.DynamicalCore(coords=("x",))
    with pytest.raises(TypeError, match="two distinct strings"):
        sw.modules.SadournyAdvection(coords=("x", "x"))


def test_core_coords_must_match_the_chart():
    with pytest.raises(ValueError,
                       match=r"DynamicalCore coords.*chart"):
        sw.Model(
            grid=sphere_grid(), csqr=CSQR,
            coriolis=sw.modules.RotationCoriolis(
                omega=(0.0, 0.0, OMEGA), coords=("lon", "lat")),
            advection=False,
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3))


def test_advection_coords_must_match_the_chart():
    grid = sphere_grid()
    with pytest.raises(ValueError,
                       match=r"SadournyAdvection coords.*chart"):
        fr.model.Model(
            grid=grid,
            modules=(
                sw.modules.DynamicalCore(
                    csqr=CSQR, coords=("lon", "lat")),
                sw.modules.SadournyAdvection()),
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3))


def test_background_flow_is_rejected_on_chart_grids():
    grid = sphere_grid()
    with pytest.raises(NotImplementedError,
                       match=r"background.*chart"):
        fr.model.Model(
            grid=grid,
            modules=(
                sw.modules.DynamicalCore(
                    csqr=CSQR, coords=("lon", "lat")),
                sw.modules.SadournyAdvection(
                    background={"u": 0.1},
                    coords=("lon", "lat"))),
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3))


def test_core_coords_property_round_trips():
    core = sw.modules.DynamicalCore(coords=("lon", "lat"))
    assert core.coords == ("lon", "lat")
    adv = sw.modules.SadournyAdvection(coords=("lon", "lat"))
    assert adv.coords == ("lon", "lat")
    assert adv.extra_halo["lon"] == 2
    assert adv.extra_halo["lat"] == 2
