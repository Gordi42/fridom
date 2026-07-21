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

ADVECT = fr.model.term_predicates.named("SadournyAdvection/advect")
GRAVITY = fr.model.term_predicates.named("Core/gravity")


def sphere_grid(nlon=2 * N, nlat=N, radius=1.0):
    """Lat-lon sphere chart grid, polar caps excluded (the recipe)."""
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=radius, lat_extent=(-LAT_MAX, LAT_MAX))


def sphere_model(grid=None, *, advection=True, omega=OMEGA,
                 csqr=CSQR, ro=RO):
    """Assemble the spherical shallow-water model (the preset)."""
    if grid is None:
        grid = sphere_grid()
    return sw.Model(
        grid=grid,
        core=sw.Core(froude_number=ro, depth=csqr,
                     coords=("lon", "lat")),
        scaling=fr.scaling.GravityWave(),
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


def test_chart_model_keeps_the_exempt_halo():
    # on a chart grid the core's gravity term resolves metric-aware
    # kinds the halo tracer cannot follow, so the module stays exempt
    # through its extra_halo declaration even without advection (the
    # flat-gate must not leak here). The width is now DERIVED from the
    # order-2 diff rows the term applies: 1 per coordinate, not the old
    # literal 2 -- the lat-lon sphere is orthogonal (diagonal
    # raise_index), so no cross-interp, and even a non-orthogonal chart
    # derives 1 (the cross-interp telescopes two-sided;
    # pressure_solver_halo.md).
    model = sphere_model(advection=False)
    core = model.module(sw.Core)
    assert core.extra_halo is not None  # exemption kept
    halo = model.grid.decomposition.halo
    assert halo["lon"] == 1
    assert halo["lat"] == 1


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
    # entry seam: physical U -> contravariant u^i (chart.py); exit
    # seam rescales the raised momentum tendency dU = sqrt(g_ii) du^i
    root_u = grid.metric(u.function_space.bare, "g_lonlon") ** 0.5
    root_v = grid.metric(v.function_space.bare, "g_latlat") ** 0.5
    u = u / root_u
    v = v / root_v
    inv_uu = grid.metric(u.function_space.bare, "inv_g_lonlon")
    inv_vv = grid.metric(v.function_space.bare, "inv_g_latlat")
    du = root_u * -(inv_uu * p.diff("lon").retag(u))
    dv = root_v * -(inv_vv * p.diff("lat").retag(v))
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
    # entry seam: physical U -> contravariant u^i (chart.py); the whole
    # vector-invariant hand form below is on the contravariant
    # intermediates, exit-rescaled dU = sqrt(g_ii) du^i
    root_u = grid.metric(u.function_space.bare, "g_lonlon") ** 0.5
    root_v = grid.metric(v.function_space.bare, "g_latlat") ** 0.5
    u = u / root_u
    v = v / root_v
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
    du = root_u * RO * (inv_uu * ((fv * q).to(u)
                                  - ekin.diff("lon").retag(u)))
    dv = root_v * RO * (inv_vv * (-(fu * q).to(v)
                                  - ekin.diff("lat").retag(v)))
    np.testing.assert_allclose(np.asarray(dz["p"].data),
                               np.asarray(dp.data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(dz["u"].data),
                               np.asarray(du.data), atol=1e-12)
    np.testing.assert_allclose(np.asarray(dz["v"].data),
                               np.asarray(dv.data), atol=1e-12)


def test_semi_discrete_energy_rate_is_machine_zero_on_the_sphere():
    # gravity + Sadourny advection conserve the PHYSICAL metric-weighted
    # thickness energy E = sum sqrt_g h U^2 / 2 + sqrt_g p^2 / 2 exactly
    # (semi-discrete; measured ~6e-16) — the sphere twin of the
    # Cartesian walled-grid gate. In physical components the energy
    # carries no g_ii (folded into U^2); the sqrt_g comes from integrate
    model = sphere_model(omega=0.0)
    set_random(model, seed=11)
    z = model.state
    dz = model.tendency(z)
    u, v, p = z["u"], z["v"], z["p"]
    du, dv, dp = dz["u"], dz["v"], dz["p"]
    h = z["csqr"].to(p) + RO * p

    def total(f):
        return float(f.integrate().data.ravel()[0])

    terms = (
        total(u * du * h.to(u)),
        total(0.5 * RO * (u * u) * dp.to(u)),
        total(v * dv * h.to(v)),
        total(0.5 * RO * (v * v) * dp.to(v)),
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
        grid=grid,
        core=sw.Core(froude_number=RO, depth=CSQR),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))
    explicit = sw.Model(
        grid=grid,
        core=sw.Core(froude_number=RO, depth=CSQR,
                     coords=("x", "y")),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO),
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
        grid=grid,
        core=sw.Core(froude_number=RO, depth=CSQR),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO),
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
    # on a flat grid the chart-native view is the identity
    assert z.chart["u"] is u
    assert z.chart["v"] is v
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
def test_chart_view_is_the_coordinate_velocity_on_the_sphere():
    # state.chart["u"] is the chart-native coordinate velocity
    # u^lon = U / sqrt(g_lonlon) = U / (a cos lat) (ruling (d))
    model = sphere_model(grid=sphere_grid(radius=2.0))
    set_random(model, seed=9)
    z = model.state
    grid = model.grid
    u, v = z["u"], z["v"]
    root_u = grid.metric(u.function_space.bare, "g_lonlon") ** 0.5
    root_v = grid.metric(v.function_space.bare, "g_latlat") ** 0.5
    np.testing.assert_allclose(np.asarray(z.chart["u"].data),
                               np.asarray((u / root_u).data),
                               atol=1e-13)
    np.testing.assert_allclose(np.asarray(z.chart.v.data),
                               np.asarray((v / root_v).data),
                               atol=1e-13)
    # destructuring order (zonal, meridional)
    cu, cv = z.chart.velocities
    assert np.array_equal(np.asarray(cu.data),
                          np.asarray(z.chart["u"].data))
    assert np.array_equal(np.asarray(cv.data),
                          np.asarray(z.chart["v"].data))
    # read-only: writes are refused
    with pytest.raises(TypeError, match="read-only"):
        z.chart["u"] = u
    with pytest.raises(AttributeError, match="read-only"):
        z.chart.u = u
    # the retired conversion properties are gone
    with pytest.raises(AttributeError):
        _ = z.u_physical
    with pytest.raises(AttributeError):
        _ = z.v_physical


def test_chart_view_is_the_identity_on_a_flat_grid():
    # on a flat grid u/v are already physical == chart-native
    grid = make_grid()
    model = sw.Model(
        grid=grid,
        core=sw.Core(froude_number=RO, depth=CSQR),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            1e-3, order=3))
    rng = np.random.default_rng(4)
    model.set_fields(u=rng.standard_normal((16, 16)),
                     v=rng.standard_normal((16, 16)))
    z = model.state
    assert z.chart["u"] is z["u"]
    assert z.chart["v"] is z["v"]


def test_solid_body_rotation_diagnostics():
    # physical u_east = w0 cos(lat) (i.e. u^lon = w0, solid body):
    # divergence exactly zero (sqrt_g lon-independent), vorticity
    # 2 w0 sin(lat) to truncation
    model = sphere_model()
    w0 = 0.4
    model.set_fields(
        u=lambda lon, lat: w0 * jnp.cos(lat) + 0.0 * lon)
    z = model.state
    assert float(np.abs(np.asarray(z.divergence.data)).max()) < 1e-13
    zeta = z.rel_vort
    lat = model.grid.evaluation_nodes(
        zeta.function_space, "lat").data
    exact = 2.0 * w0 * np.sin(np.asarray(lat))
    err = np.abs(np.asarray(zeta.data) - exact).max()
    assert err < 5e-2  # O(dlat^2) at nlat = 8


def test_ekin_diagnostic_is_the_physical_quadratic_on_the_sphere():
    # the state components are physical, so ekin collapses to the flat
    # quadratic 0.5 (U^2 + V^2) at centre — no metric root (D4)
    model = sphere_model()
    set_random(model, seed=13)
    z = model.state
    center = z["p"].function_space
    u_c = z["u"].to(center)
    v_c = z["v"].to(center)
    expected = 0.5 * (u_c.data**2 + v_c.data**2)
    got = model.diagnostics.ekin()
    np.testing.assert_allclose(np.asarray(got.data),
                               np.asarray(expected), atol=1e-13)


# ================================================================
#  Taught errors
# ================================================================
def test_core_rejects_bad_coords():
    # coords are validated before the kwarg-set check, so no physics
    # kwargs are needed to reach the taught error
    with pytest.raises(TypeError, match="two distinct strings"):
        sw.Core(coords=("x",))
    with pytest.raises(TypeError, match="two distinct strings"):
        sw.modules.SadournyAdvection(coords=("x", "x"))


def test_core_coords_must_match_the_chart():
    with pytest.raises(ValueError,
                       match=r"sw\.Core coords.*chart"):
        sw.Model(
            grid=sphere_grid(),
            core=sw.Core(froude_number=1.0, depth=CSQR),
            scaling=fr.scaling.GravityWave(),
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
                sw.Core(froude_number=1.0, depth=CSQR,
                        coords=("lon", "lat")),
                sw.modules.SadournyAdvection()),
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3),
            scaling=fr.scaling.GravityWave())


def test_background_flow_is_rejected_on_chart_grids():
    grid = sphere_grid()
    with pytest.raises(NotImplementedError,
                       match=r"background.*chart"):
        fr.model.Model(
            grid=grid,
            modules=(
                sw.Core(froude_number=1.0, depth=CSQR,
                        coords=("lon", "lat")),
                sw.modules.SadournyAdvection(
                    background={"u": 0.1},
                    coords=("lon", "lat"))),
            time_stepper=fr.model.time_steppers.AdamBashforth(
                1e-3, order=3),
            scaling=fr.scaling.GravityWave())


def test_core_coords_property_round_trips():
    core = sw.Core(froude_number=1.0, coords=("lon", "lat"))
    assert core.coords == ("lon", "lat")
    adv = sw.modules.SadournyAdvection(coords=("lon", "lat"))
    assert adv.coords == ("lon", "lat")
    assert adv.extra_halo["lon"] == 2
    assert adv.extra_halo["lat"] == 2
