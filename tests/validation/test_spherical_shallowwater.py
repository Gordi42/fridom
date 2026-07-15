"""
Spherical shallow water (coordinate-systems plan, stage C2).

The model half of stage C2: the unchanged shallowwater2 modules on a
lat-lon sphere chart (periodic lon x bounded lat, polar caps
excluded), prognosing contravariant velocity components (the
convention recorded in ``shallowwater2/modules/core.py``):

- **Flat limit**: on the identity chart (X = (x, y)) the metric-aware
  paths reproduce the chartless Cartesian model exactly — every metric
  factor is an exact 1.0 and each extra multiplication/division is by
  1.0, so the two evaluate the same floating-point expression and agree
  **bitwise** unfused (asserted under ``jax.disable_jit``). Under jit
  they agree to a few ULP, not bitwise: the chart is a structurally
  different program (two extra neutral combines), and XLA fuses and
  FMA-contracts it differently — see the comment in the test.
- **Mass**: the flux-form metric divergence + structural cap
  impermeability conserve the sqrt(g)-weighted mass to rounding over
  600 steps (measured ~1e-16 relative).
- **TC2**: Williamson-style steady zonal geostrophic balance adapted
  to the capped sphere stays steady to truncation error — measured
  (400 steps, dt 2e-3): p L2 error 5.4e-3 / 1.3e-3 and max v
  3.4e-3 / 9.8e-4 at 32x16 / 64x32 (2nd-order convergence).
- **Energy**: the semi-discrete rate of gravity + Sadourny advection
  under the metric thickness-weighted energy is machine zero (unit
  suite); here the AB3 run-level drift stays bounded (measured
  7.2e-5 relative over 600 steps).
- **Polar caps**: no-normal-flow is structural (the wall face is a
  boundary condition, not a DOF) and exact.
- Plus the C2 gates: compile-once across steps and forced-4
  device-count invariance (the ``test_chart_manifolds`` precedent).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

TWO_PI = float(2.0 * np.pi)
LAT_MAX = float(np.deg2rad(80.0))

#: TC2 parameters (nondimensional unit sphere, Ro = 1)
OMEGA, U0, GH0 = 2.0, 0.2, 2.0


def sphere_grid(nlon, nlat, radius=1.0, device_ids=None):
    """Lat-lon sphere chart grid (the documented sw.Model recipe)."""
    mlon = fr.spatial.meshes.IntervalMesh(nlon, (0.0, TWO_PI),
                                          name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(
        nlat, (-LAT_MAX, LAT_MAX), periodic=False, name="lat")
    mapping = fr.spatial.CoordinateMapping(chart={
        "X": lambda lon, lat: (
            radius * jnp.cos(lat) * jnp.cos(lon),
            radius * jnp.cos(lat) * jnp.sin(lon),
            radius * jnp.sin(lat))}, orthogonal=True)
    return fr.spatial.Grid((mlon, mlat), mapping=mapping,
                           device_ids=device_ids)


def sphere_model(nlon=32, nlat=16, *, csqr=GH0, ro=1.0,
                 omega=OMEGA, dt=2e-3, device_ids=None):
    """Assemble the spherical shallow-water preset."""
    return sw.Model(
        grid=sphere_grid(nlon, nlat, device_ids=device_ids),
        coords=("lon", "lat"), csqr=csqr, rossby_number=ro,
        coriolis=sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, omega), coords=("lon", "lat"),
            metric_weight="csqr"),
        time_stepper=fr.model.time_steppers.AdamBashforth(
            dt, order=3))


def h_energy(model):
    """Evaluate the metric thickness-weighted energy invariant.

    ``E = int [ g_ii h (u^i)^2 / 2 + p^2 / 2 ]`` with
    ``h = c^2 + Ro p`` — the **public** diagnostic
    (``sw.diagnostics.etot_full``, one implementation for the scheme
    and the user); ``integrate`` carries the sqrt(g) area element on
    chart grids. The machine-zero semi-discrete production rate of
    this functional under gravity + Sadourny is pinned in
    ``tests/shallowwater2/test_diagnostics.py`` (sphere and flat).
    """
    return model.diagnostics.etot_full().integrate().item()


def swirl(lon, lat):
    """Return a smooth cap-respecting pressure perturbation."""
    return (0.05 * jnp.cos(lat) ** 2 * jnp.sin(2.0 * lon)
            + 0.03 * jnp.sin(lat) * jnp.cos(lat) ** 2
            * jnp.cos(lon))


# ================================================================
#  Flat limit: identity chart == chartless Cartesian, bitwise
# ================================================================
def test_identity_chart_run_is_bitwise_flat():
    # the f-plane as a DERIVED special case: on the identity chart
    # X = (x, y, 0) the surface normal is exactly (0, 0, 1), so
    # RotationCoriolis(omega=(0, 0, 0.5)) derives f = 2 Omega . n = 1
    # and its metric factors (sqrt_g = g_xx = g_yy = 1) are exact
    # ones — the chart run must reproduce the chartless Cartesian
    # f-plane run BITWISE, tendency and 50 steps.
    n = 16

    def meshes():
        return (fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                               name="x"),
                fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                               name="y"))

    mx, my = meshes()
    chart_grid = fr.spatial.Grid(
        (mx, my), mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x, y, 0.0 * x)},
            orthogonal=True))
    fx, fy = meshes()
    flat_grid = fr.spatial.Grid((fx, fy))

    def build(grid, coriolis, **kwargs):
        return sw.Model(
            grid=grid, csqr=0.7, rossby_number=0.4,
            coriolis=coriolis,
            time_stepper=fr.model.time_steppers.AdamBashforth(
                2e-3, order=3), **kwargs)

    chart = build(chart_grid, sw.modules.RotationCoriolis(
        omega=(0.0, 0.0, 0.5), coords=("x", "y")), coords=("x", "y"))
    flat = build(flat_grid, sw.modules.FPlaneCoriolis(f0=1.0))
    assert float(chart.state["f_coriolis"].data.min()) == 1.0
    assert float(chart.state["f_coriolis"].data.max()) == 1.0
    rng = np.random.default_rng(7)
    fields = {
        "u": rng.standard_normal((n, n)),
        "v": rng.standard_normal((n, n)),
        "p": 0.3 * rng.standard_normal((n, n))}
    chart.set_fields(**fields)
    flat.set_fields(**fields)

    # The chart path evaluates the *same floating-point expression* as
    # the flat path: every metric factor is an exact 1.0 and each extra
    # multiplication/division is by 1.0. With jit disabled — op by op,
    # no XLA fusion — the two agree BITWISE. That is the invariant this
    # test exists to pin, and it is asserted exactly.
    with jax.disable_jit():
        dzc = chart.tendency(chart.state)
        dzf = flat.tendency(flat.state)
        for name in ("u", "v", "p"):
            assert np.array_equal(np.asarray(dzc[name].data),
                                  np.asarray(dzf[name].data))

    # Under jit the two are no longer *bitwise* equal, and cannot be
    # asked to be. The chart graph carries two extra (algebraically
    # neutral) metric combines, so it is a structurally different
    # program; since the storage-frame field arithmetic removed the
    # pad/unpad round trips that used to sit between field operations
    # as fusion barriers, XLA now fuses long elementwise chains and
    # contracts multiply+add pairs into FMAs — and it makes those
    # choices differently for the 32-op chart chain than for the 30-op
    # flat chain. A contracted FMA rounds differently from a separate
    # multiply-then-add, so the momentum tendencies differ in the last
    # bit (measured: 1 ULP, 1.3e-16 relative; p is still exact).
    # Demanding bitwise agreement here would forbid the compiler from
    # fusing two structurally different programs differently, i.e. it
    # would forbid the optimization rather than test the physics.
    dzc = chart.tendency(chart.state)
    dzf = flat.tendency(flat.state)
    for name in ("u", "v", "p"):
        assert np.allclose(np.asarray(dzc[name].data),
                           np.asarray(dzf[name].data),
                           rtol=1e-14, atol=1e-14)

    chart.advance(50)
    flat.advance(50)
    for name in ("u", "v", "p"):
        assert np.allclose(np.asarray(chart.state[name].data),
                           np.asarray(flat.state[name].data),
                           rtol=1e-12, atol=1e-12)


# ================================================================
#  Global mass conservation on the sphere
# ================================================================
def test_global_mass_is_conserved_to_rounding():
    model = sphere_model()
    # offset the perturbation so the global mass is O(1) and the
    # relative drift is meaningful
    model.set_fields(
        p=lambda lon, lat: 0.5 + swirl(lon, lat),
        u=lambda lon, lat: 0.05 * jnp.sin(lat) ** 2 + 0.0 * lon)
    mass0 = float(model.state["p"].integrate().data.ravel()[0])
    model.advance(600)
    mass1 = float(model.state["p"].integrate().data.ravel()[0])
    assert abs(mass1 - mass0) / abs(mass0) < 1e-13


# ================================================================
#  TC2: steady zonal geostrophic balance (capped sphere)
# ================================================================
def tc2_errors(nlon, nlat, steps=400, dt=2e-3):
    """Run the balanced solid-body state; return drift norms."""
    model = sphere_model(nlon, nlat, dt=dt)
    # balanced pair: u^lon = U0 (contravariant solid body, i.e.
    # u_east = U0 cos(lat)), v = 0, and the height field
    # p = -(Omega U0 + U0^2/2) sin^2(lat) (unit radius; gradient
    # balance of Coriolis + metric self-advection)
    model.set_fields(
        u=lambda lon, lat: U0 + 0.0 * lon + 0.0 * lat,
        p=lambda lon, lat: -(OMEGA * U0 + 0.5 * U0 ** 2)
        * jnp.sin(lat) ** 2 + 0.0 * lon)
    p0 = np.asarray(model.state["p"].data).copy()
    u0 = np.asarray(model.state["u"].data).copy()
    scale = np.abs(p0).max()
    model.advance(steps)
    p_err = float(np.sqrt(np.mean(
        (np.asarray(model.state["p"].data) - p0) ** 2)) / scale)
    u_err = float(np.abs(
        np.asarray(model.state["u"].data) - u0).max() / U0)
    v_max = float(np.abs(
        np.asarray(model.state["v"].data)).max() / U0)
    return p_err, u_err, v_max


def test_tc2_stays_steady_and_converges():
    # measured (400 steps, dt 2e-3): p L2 5.4e-3 / 1.3e-3,
    # u max 1.4e-2 / 3.4e-3, v max 3.4e-3 / 9.8e-4 at
    # 32x16 / 64x32 — the truncation-level drift of the discretely
    # unbalanced projection, converging at 2nd order
    coarse = tc2_errors(32, 16)
    assert coarse[0] < 1.5e-2   # p L2
    assert coarse[1] < 3.0e-2   # u max
    assert coarse[2] < 1.0e-2   # v max
    fine = tc2_errors(64, 32)
    assert fine[0] < 0.45 * coarse[0]
    assert fine[1] < 0.45 * coarse[1]
    assert fine[2] < 0.45 * coarse[2]


# ================================================================
#  Energy: machine-zero semi-discrete rate (unit suite) + bounded
#  AB3 run-level drift of the exact invariant
# ================================================================
def test_energy_drift_stays_bounded_over_a_long_run():
    ro = 1.0
    model = sphere_model(ro=ro)
    model.set_fields(
        p=swirl,
        u=lambda lon, lat: 0.05 * jnp.sin(lat) ** 2 + 0.0 * lon)
    e0 = h_energy(model)
    peak = 0.0
    for _ in range(12):
        model.advance(50)              # 600 steps
        peak = max(peak, abs(h_energy(model) - e0) / e0)
    for name in ("u", "v", "p"):
        assert not bool(model.state[name].has_nan())
    # measured 7.2e-5: pure AB3 time-integration error (the
    # semi-discrete rate is machine zero, tests/shallowwater2)
    assert peak < 5e-4


# ================================================================
#  Polar caps: no-normal-flow is structural and exact
# ================================================================
def test_no_normal_flow_at_the_polar_caps():
    model = sphere_model()
    # the wall-normal velocity carries interior faces only: the cap
    # value is a boundary condition (an exact zero), not a DOF —
    # there is nothing to drift
    v_space = model.state["v"].function_space
    assert model.state["v"].shape == (32, 15)
    assert v_space.bare.factor("lat").node_set.name == "INNER"
    assert not v_space.bare.factor("lat").bc.is_free
    model.set_fields(p=lambda lon, lat: 0.5 + swirl(lon, lat))
    model.advance(200)
    # impermeability closes the mass budget exactly and the state
    # spaces (hence the structural wall) are preserved
    assert model.state["v"].function_space is v_space
    dp = model.tendency(model.state)["p"]
    rate = float(dp.integrate().data.ravel()[0])
    scale = float(abs(dp).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-14


# ================================================================
#  Gates: compile-once, device-count invariance
# ================================================================
def test_spherical_advance_compiles_once(compile_counter):
    model = sphere_model(16, 8)
    model.set_fields(p=swirl)
    model.advance(5)                    # warm every path
    compile_counter.reset()
    model.advance(5)
    assert compile_counter.count == 0


@pytest.mark.multi_device
def test_spherical_tendency_is_device_count_invariant():
    # NOT bitwise, unlike the eager chart operators
    # (test_chart_manifolds) and the flat Cartesian model (both
    # verified bitwise): the jitted tendency's metric-scaled stencil
    # chains give XLA mul+add patterns whose FMA/fusion contraction
    # depends on the per-shard array shapes, so 1-device and
    # 4-device compilations round the last ulp differently. The
    # sharded metric fields themselves ARE bitwise identical
    # (verified); the measured tendency deviation is <= 1.7e-16
    # absolute on O(0.1..1) tendencies — asserted at 1e-13 as the
    # honest to-rounding bound.
    def run(device_ids):
        model = sphere_model(32, 16, device_ids=device_ids)
        model.set_fields(
            p=swirl,
            u=lambda lon, lat: 0.05 * jnp.sin(lat) ** 2 + 0.0 * lon)
        dz = model.tendency(model.state)
        return {name: np.asarray(dz[name].data)
                for name in ("u", "v", "p")}

    many = run(None)
    one = run((0,))
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(many[name], one[name],
                                   rtol=0.0, atol=1e-13)
