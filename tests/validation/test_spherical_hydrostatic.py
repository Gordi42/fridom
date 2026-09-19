r"""
Spherical hydrostatic model (spherical-models plan, phases S1 + S2).

The 3-D hydrostatic model on the thin-shell ``(lon, lat, z)`` sphere
(periodic lon x bounded lat with the polar caps excluded x flat
bounded z) and on the torus, in the **orthogonal thin-shell** metric
form: area-weighted transports, one ``1/sqrt_g`` division, the momentum
curvature source, the physical pressure gradient ``d_i p / h_i`` and
the explicit free surface. The stored ``u``, ``v`` are the physical
(m/s) components on every grid. Gates (numbers measured on CPU,
float64):

- **Identity-chart reduction (S1-1 / S2-1)**: on ``X = (x, y, 0)`` the
  whole chart program (advection + core + free surface + rotation)
  equals the chartless Cartesian model **bitwise** unfused
  (``jax.disable_jit``) and to a few ULP jitted (measured: exactly 0
  jitted too) — tendency and a 20-step run.
- **TC2 / sw2 cross-check (S1-2 / S2-2 / S2-4)**: Williamson-style
  solid-body rotation in geostrophic balance on the capped sphere, a
  z-uniform two-level column (the one-layer limit; a single z cell is
  below the two-cell storage halo). Measured (400 steps, dt 2e-3):
  ps L2 2.08e-3 / 5.09e-4, max v 7.2e-4 / 2.0e-4 at 32x16 / 64x32 —
  2nd-order convergence, and the same truncation-level drift as the
  vector-invariant ``sw2`` reference on the same grid (p L2 2.044e-3
  at 32x16; asserted within 50 %, measured within 2 %).
- **Rossby-Haurwitz wave vs sw2**: a wave-4 Rossby-Haurwitz wave on a
  deep layer (``g H = 50``, so the linear free surface and the
  nonlinear shallow-water thickness differ at ``O(p'/gH)``) run in
  both models: while the wave's ``v`` changes by 31 %, the flux-form
  hydrostatic model and the vector-invariant ``sw2`` agree to 6.3e-3
  (32x16) and 1.7e-3 (64x32) relative — two independent
  discretizations of the metric terms converging onto each other at
  2nd order (meridional advection and both curvature terms are O(1)
  here, unlike in the zonal TC2 state).
- **Curvature discriminator (S1-3)**: with the curvature source
  switched off the same state drifts at O(1) and does **not** converge
  (measured ps L2 9.8e-3 / 8.1e-3).
- **Tracer (S1-4)**: the flux form conserves the sqrt(g)-weighted
  tracer content to rounding (surface closure off) and preserves a
  constant exactly (surface closure on, the default).
- **Rest states (S2-3)**: a stratified fluid at rest stays at rest to
  machine precision on the sphere and on the torus.
- **Caps / volume (S2-5)**: the cap-normal velocity is structural (no
  DOF) and the sqrt(g)-weighted ``ps`` budget closes to rounding.
- **Autodiff (S1-5 / S2-6)**: ``jax.grad`` through a short spherical
  run matches a central finite difference (rtol 1e-4).
- **Fences (S1-7 / S2-7)**: upwind / WENO advection and the implicit /
  split-explicit free surfaces stay taught refusals on a chart.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
import fridom.shallowwater2 as sw
from fridom.model.params import TIME_STEP
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh
CenteredAdvection = fr.model.modules.CenteredAdvection
RotationCoriolis = fr.model.modules.RotationCoriolis

LAT_MAX = float(np.deg2rad(80.0))
HOR = ("lon", "lat")

#: TC2 parameters (unit sphere; the sw2 validation's own numbers)
OMEGA, U0, GH0 = 2.0, 0.2, 2.0


def sphere_grid(nlon, nlat, nz, depth=1.0, device_ids=None):
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=1.0, lat_extent=(-LAT_MAX, LAT_MAX),
        vertical=IM(nz, (-depth, 0.0), periodic=False, name="z"),
        device_ids=device_ids)


def sphere_model(nlon=32, nlat=16, nz=2, *, dt=2e-3, buoyancy=False,
                 advection=None, device_ids=None, extra=()):
    """Assemble the spherical hydrostatic model (explicit surface)."""
    if advection is None:
        advection = CenteredAdvection()
    return hy.Model(
        modules_extra=list(extra),
        grid=sphere_grid(nlon, nlat, nz, device_ids=device_ids),
        core=hy.Core(gravity=GH0, horizontal=HOR),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=RotationCoriolis((0.0, 0.0, OMEGA)),
        buoyancy=hy.BuoyancyTracer() if buoyancy else None,
        free_surface=hy.ExplicitFreeSurface(horizontal=HOR),
        advection=advection)


def nodes(model, name, coord):
    field = model.state[name]
    return np.asarray(model.grid.evaluation_nodes(
        field.function_space, coord).data)


def set_tc2(model, pname="ps"):
    """Set the steady solid-body state; return (u0, p0, area weight)."""
    state = model.state
    u0 = U0 * np.cos(nodes(model, "u", "lat")) * np.ones(state["u"].shape)
    lat_c = nodes(model, pname, "lat")
    p0 = (-(OMEGA * U0 + 0.5 * U0 ** 2) * np.sin(lat_c) ** 2
          * np.ones(state[pname].shape))
    model.set_fields(**{"u": u0, pname: p0})
    return u0, p0, np.cos(lat_c) * np.ones(state[pname].shape)


def tc2_errors(model, pname="ps", steps=400):
    u0, p0, area = set_tc2(model, pname)
    model.advance(steps)
    assert not model.panicked
    state = model.state
    err = np.asarray(state[pname].data) - p0
    p_l2 = float(np.sqrt((area * err ** 2).sum() / area.sum()))
    u_max = float(np.abs(np.asarray(state["u"].data) - u0).max())
    v_max = float(np.abs(np.asarray(state["v"].data)).max())
    return p_l2, u_max, v_max


def area_content(field):
    """Return the sqrt(g)-weighted sum of a field (its chart content)."""
    sqrt_g = np.asarray(field.grid.metric(
        field.function_space.bare, "sqrt_g").data)
    return float((sqrt_g * np.asarray(field.data)).sum())


# ================================================================
#  Flat limit: identity chart == chartless Cartesian, bitwise
# ================================================================
def test_identity_chart_run_is_bitwise_flat():
    def meshes():
        return (IM(8, (0.0, 1.0), name="x"),
                IM(8, (0.0, 1.0), periodic=False, name="y"),
                IM(4, (-1.0, 0.0), periodic=False, name="z"))

    def build(grid, coriolis):
        return hy.Model(
            grid=grid, core=hy.Core(gravity=1.0),
            time_stepper=AdamBashforth(1e-3, order=3),
            coriolis=coriolis, buoyancy=hy.BuoyancyTracer(),
            free_surface=hy.ExplicitFreeSurface(),
            advection=CenteredAdvection())

    chart = build(
        fr.spatial.Grid(meshes(), mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x, y, 0.0 * x)},
            orthogonal=True)),
        RotationCoriolis((0.0, 0.0, 0.5), coords=("x", "y")))
    flat = build(fr.spatial.Grid(meshes()), hy.FPlaneCoriolis(f0=1.0))
    assert chart.grid.chart_coords == ("x", "y")
    rng = np.random.default_rng(3)
    names = ("u", "v", "b", "ps")
    fields = {k: 0.1 * rng.standard_normal(flat.state[k].shape)
              for k in names}
    chart.set_fields(**fields)
    flat.set_fields(**fields)
    # every metric factor is an exact 1.0 and every scale-factor
    # derivative an exact 0.0, so op by op the chart program evaluates
    # the flat floating-point expression: bitwise, asserted exactly
    with jax.disable_jit():
        dzc = chart.tendency(chart.state)
        dzf = flat.tendency(flat.state)
        for name in names:
            assert np.array_equal(np.asarray(dzc[name].data),
                                  np.asarray(dzf[name].data))
    # jitted: a structurally different program may fuse differently
    # (the sw2 precedent) — a few ULP (measured: exactly equal)
    dzc = chart.tendency(chart.state)
    dzf = flat.tendency(flat.state)
    chart.advance(20)
    flat.advance(20)
    for name in names:
        assert np.allclose(np.asarray(dzc[name].data),
                           np.asarray(dzf[name].data),
                           rtol=1e-13, atol=1e-15)
        assert np.allclose(np.asarray(chart.state[name].data),
                           np.asarray(flat.state[name].data),
                           rtol=1e-12, atol=1e-14)


# ================================================================
#  TC2: steadiness, 2nd-order convergence, the sw2 cross-check
# ================================================================
@pytest.fixture(scope="module")
def tc2_pair():
    return (tc2_errors(sphere_model(32, 16)),
            tc2_errors(sphere_model(64, 32)))


def test_tc2_stays_steady_and_converges(tc2_pair):
    # measured: ps L2 2.08e-3 / 5.09e-4, u max 1.6e-3 / 3.9e-4,
    # v max 7.2e-4 / 2.0e-4 at 32x16 / 64x32 (ratio ~4: 2nd order)
    coarse, fine = tc2_pair
    assert coarse[0] < 4.0e-3
    assert coarse[1] < 3.0e-3
    assert coarse[2] < 1.5e-3
    for c, f in zip(coarse, fine, strict=True):
        assert f < 0.3 * c


def test_tc2_matches_the_shallow_water_reference(tc2_pair):
    # the metric-correct reference: vector-invariant sw2 on the same
    # sphere, same state, same steps. The z-uniform hydrostatic column
    # is its one-layer limit (linear free surface): both hold the
    # balance to the same truncation-level drift (measured p L2
    # 2.076e-3 here vs 2.044e-3 for sw2 — within 2 %)
    reference = sw.Model(
        advection=sw.SadournyAdvection(),
        grid=fr.spatial.spherical.Grid(
            (32, 16), radius=1.0, lat_extent=(-LAT_MAX, LAT_MAX)),
        core=sw.Core(gravity=1.0, depth=GH0, coords=HOR),
        coriolis=sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, OMEGA), coords=HOR, metric_weight="csqr"),
        time_stepper=AdamBashforth(2e-3, order=3))
    ref = tc2_errors(reference, "p")
    coarse, _ = tc2_pair
    for mine, theirs in zip(coarse, ref, strict=True):
        assert 0.5 * theirs < mine < 1.5 * theirs


# ================================================================
#  Rossby-Haurwitz wave: the hydrostatic column tracks sw2
# ================================================================
RH_W, RH_K, RH_R, RH_GH = 0.1, 0.1, 4, 50.0


def _rh_u(lon, lat):
    c, s = jnp.cos(lat), jnp.sin(lat)
    return RH_W * c + RH_K * c ** (RH_R - 1) * (
        RH_R * s ** 2 - c ** 2) * jnp.cos(RH_R * lon)


def _rh_v(lon, lat):
    return (-RH_K * RH_R * jnp.cos(lat) ** (RH_R - 1) * jnp.sin(lat)
            * jnp.sin(RH_R * lon))


def _rh_p(lon, lat):
    r, c = RH_R, jnp.cos(lat)
    a = (0.5 * RH_W * (2 * OMEGA + RH_W) * c ** 2
         + 0.25 * RH_K ** 2 * c ** (2 * r) * (
             (r + 1) * c ** 2 + (2 * r ** 2 - r - 2)
             - 2 * r ** 2 / c ** 2))
    b = (2 * (OMEGA + RH_W) * RH_K / ((r + 1) * (r + 2)) * c ** r
         * ((r ** 2 + 2 * r + 2) - (r + 1) ** 2 * c ** 2))
    cc = 0.25 * RH_K ** 2 * c ** (2 * r) * ((r + 1) * c ** 2 - (r + 2))
    return a + b * jnp.cos(r * lon) + cc * jnp.cos(2 * r * lon)


def _rh_pair(nlon, nlat, dt, steps):
    """Run the wave in both models; return (v change, v mismatch)."""
    stepper = AdamBashforth(dt, order=3)
    hydro = hy.Model(
        grid=sphere_grid(nlon, nlat, 2),
        core=hy.Core(gravity=RH_GH, horizontal=HOR),
        time_stepper=stepper,
        coriolis=RotationCoriolis((0.0, 0.0, OMEGA)),
        free_surface=hy.ExplicitFreeSurface(horizontal=HOR),
        advection=CenteredAdvection())
    hydro.set_fields(
        u=lambda lon, lat, z: _rh_u(lon, lat) + 0.0 * z,
        v=lambda lon, lat, z: _rh_v(lon, lat) + 0.0 * z, ps=_rh_p)
    shallow = sw.Model(
        advection=sw.SadournyAdvection(),
        grid=fr.spatial.spherical.Grid(
            (nlon, nlat), radius=1.0, lat_extent=(-LAT_MAX, LAT_MAX)),
        core=sw.Core(gravity=1.0, depth=RH_GH, coords=HOR),
        coriolis=sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, OMEGA), coords=HOR, metric_weight="csqr"),
        time_stepper=AdamBashforth(dt, order=3))
    shallow.set_fields(u=_rh_u, v=_rh_v, p=_rh_p)
    v_start = np.asarray(shallow.state["v"].data).copy()
    hydro.advance(steps)
    shallow.advance(steps)
    assert not hydro.panicked
    assert not shallow.panicked
    v_hy = np.asarray(hydro.state["v"].data)[..., 0]
    v_sw = np.asarray(shallow.state["v"].data)

    def norm(x):
        return float(np.sqrt((x ** 2).mean()))

    return (norm(v_sw - v_start) / norm(v_start),
            norm(v_hy - v_sw) / norm(v_sw))


def test_rossby_haurwitz_wave_tracks_the_shallow_water_reference():
    # measured: the wave's v changes by 31 % over the run while the two
    # models differ by 6.3e-3 (32x16) / 1.7e-3 (64x32)
    change, coarse = _rh_pair(32, 16, 1e-3, 2000)
    _, fine = _rh_pair(64, 32, 5e-4, 4000)
    assert change > 0.2          # the comparison is not a still life
    assert coarse < 1.5e-2
    assert fine < 0.4 * coarse   # the two schemes converge together


def test_curvature_source_is_load_bearing(monkeypatch, tc2_pair):
    # drop the u v tan(lat) / a source: the solid-body state drifts at
    # O(U0^2) and does NOT converge (measured 9.8e-3 / 8.1e-3)
    monkeypatch.setattr(CenteredAdvection, "_curvature",
                        lambda self, state, qname: None)  # noqa: ARG005
    coarse = tc2_errors(sphere_model(32, 16))
    fine = tc2_errors(sphere_model(64, 32))
    good_coarse, good_fine = tc2_pair
    assert coarse[0] > 3.0 * good_coarse[0]
    assert fine[0] > 10.0 * good_fine[0]
    assert fine[0] > 0.6 * coarse[0]


# ================================================================
#  Tracer: sqrt(g)-weighted conservation, constancy
# ================================================================
def _stirred(b0, *, surface_flux):
    model = sphere_model(32, 16, 4, buoyancy=True,
                         advection=CenteredAdvection(
                             surface_flux=surface_flux))
    state = model.state
    rng = np.random.default_rng(1)
    model.set_fields(
        u=0.2 * rng.standard_normal(state["u"].shape),
        v=0.2 * rng.standard_normal(state["v"].shape),
        ps=0.05 * rng.standard_normal(state["ps"].shape),
        b=b0(model))
    return model


def test_tracer_content_is_conserved_to_rounding():
    def b0(model):
        lon, lat, z = (nodes(model, "b", c) for c in ("lon", "lat", "z"))
        return 1e-2 * np.cos(lat) ** 2 * np.sin(2 * lon) * (1 + z)

    model = _stirred(b0, surface_flux=False)
    before = area_content(model.state["b"])
    scale = area_content(abs(model.state["b"]))
    ps_before = area_content(model.state["ps"])
    ps_scale = area_content(abs(model.state["ps"]))
    model.advance(200)
    assert not model.panicked
    # measured -3.5e-17 (tracer), -5.0e-17 (ps volume)
    assert abs(area_content(model.state["b"]) - before) < 1e-13 * scale
    assert (abs(area_content(model.state["ps"]) - ps_before)
            < 1e-13 * ps_scale)


def test_constant_tracer_stays_constant():
    model = _stirred(lambda m: 0.37 * np.ones(m.state["b"].shape),
                     surface_flux=None)
    model.advance(200)
    b = np.asarray(model.state["b"].data)
    assert np.abs(b - 0.37).max() < 1e-13


# ================================================================
#  Rest states: sphere and torus, machine precision
# ================================================================
def _torus_grid():
    return fr.spatial.Grid(
        (IM(16, (0.0, 2 * np.pi), name="tor"),
         IM(12, (0.0, 2 * np.pi), name="pol"),
         IM(6, (-1.0, 0.0), periodic=False, name="z")),
        mapping=fr.spatial.charts.torus(2.0, 0.7))


@pytest.mark.parametrize(("make_grid", "hor"), [
    pytest.param(lambda: sphere_grid(16, 8, 6), HOR, id="sphere"),
    pytest.param(_torus_grid, ("tor", "pol"), id="torus"),
])
def test_stratified_rest_state_stays_at_rest(make_grid, hor):
    grid = make_grid()
    model = hy.Model(
        grid=grid, core=hy.Core(gravity=GH0, horizontal=hor),
        time_stepper=AdamBashforth(2e-3, order=3),
        coriolis=RotationCoriolis((0.0, 0.0, OMEGA), coords=hor),
        buoyancy=hy.BuoyancyTracer(),
        free_surface=hy.ExplicitFreeSurface(horizontal=hor),
        advection=CenteredAdvection())
    z = nodes(model, "b", "z")
    model.set_fields(b=np.exp(2 * z) * np.ones(model.state["b"].shape))
    model.advance(50)
    # measured ~1e-17 (the rounding of the p_hyd column sums)
    for name in ("u", "v", "w", "ps"):
        assert np.abs(np.asarray(model.state[name].data)).max() < 1e-14


# ================================================================
#  Polar caps: structural no-normal-flow, closed volume budget
# ================================================================
def test_no_normal_flow_at_the_polar_caps():
    model = sphere_model(32, 16, 4)
    v_space = model.state["v"].function_space
    assert model.state["v"].shape == (32, 15, 4)
    assert v_space.bare.factor("lat").node_set.name == "INNER"
    assert not v_space.bare.factor("lat").bc.is_free
    rng = np.random.default_rng(5)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape))
    rate = model.tendency(model.state)["ps"]
    assert (abs(area_content(rate))
            < 1e-13 * area_content(abs(rate)))


# ================================================================
#  Autodiff: grad through a short spherical run vs central FD
# ================================================================
def test_grad_through_a_spherical_run_matches_fd():
    model = sphere_model(16, 8, 4, buoyancy=True)
    lon, lat, z = (nodes(model, "b", c) for c in ("lon", "lat", "z"))
    set_tc2(model)
    model.set_fields(
        b=1e-2 * np.cos(lat) ** 2 * np.sin(lon) * (1 + z))
    run = model.propagator(wrt=(TIME_STEP,), steps=8)

    def loss(dt):
        state = run((dt,)).state
        return sum(jnp.sum(state[name].data ** 2)
                   for name in ("u", "v", "b", "ps"))

    dt0 = jnp.asarray(2e-3)
    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    eps = 2e-7
    fd = (float(loss(dt0 + eps)) - float(loss(dt0 - eps))) / (2 * eps)
    assert grad == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Forced-4: device-count invariance (S1-6 / S2-6)
# ================================================================
@pytest.mark.multi_device
def test_spherical_tendency_is_device_count_invariant():
    # to rounding, not bitwise (the sw2 precedent): the metric-scaled
    # stencil chains fuse / FMA-contract per shard shape
    def run(device_ids):
        model = sphere_model(32, 16, 4, buoyancy=True,
                             device_ids=device_ids)
        rng = np.random.default_rng(11)
        model.set_fields(**{
            k: 0.1 * rng.standard_normal(model.state[k].shape)
            for k in ("u", "v", "b", "ps")})
        dz = model.tendency(model.state)
        return {name: np.asarray(dz[name].data)
                for name in ("u", "v", "b", "ps")}

    many = run(None)
    one = run((0,))
    for name in ("u", "v", "b", "ps"):
        np.testing.assert_allclose(many[name], one[name],
                                   rtol=0.0, atol=1e-12)


# ================================================================
#  Surface forcing on the flat vertical of the chart
# ================================================================
def test_wind_stress_drives_the_top_cell_on_the_sphere():
    # the wall weight 1/dz along the unmapped vertical is metric-free
    # (thin shell), so WindStress binds on the chart and accelerates
    # exactly the top cell row at tau / dz
    tau, nz = 3e-3, 4
    model = sphere_model(
        16, 8, nz, advection=None,
        extra=[fr.model.modules.WindStress(
            tau_x=lambda lon, lat: tau * jnp.cos(lat) + 0.0 * lon)])
    du = model.tendency(model.state)["u"]
    lat = np.asarray(model.grid.evaluation_nodes(
        du.function_space, "lat").data)
    got = np.asarray(du.data)
    exact_top = tau * np.cos(lat[..., 0]) * nz + 0.0 * got[..., -1]
    assert np.allclose(got[..., -1], exact_top, rtol=1e-12)
    assert np.abs(got[..., :-1]).max() == 0.0


# ================================================================
#  Fences that must remain
# ================================================================
@pytest.mark.parametrize("scheme", [
    pytest.param(lambda: fr.model.modules.UpwindAdvection(order=3),
                 id="upwind"),
    pytest.param(lambda: fr.model.modules.WENOAdvection(order=5),
                 id="weno"),
])
def test_biased_advection_on_the_sphere_is_a_taught_error(scheme):
    with pytest.raises(NotImplementedError, match="metric-blind"):
        sphere_model(16, 8, 4, advection=scheme())
