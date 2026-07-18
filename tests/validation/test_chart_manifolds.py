"""
Chart-embedded manifolds (coordinate-systems plan, stage C2).

The metric-aware vector-calculus entries on chart-coupled grids,
validated on the two CS-D1 reference manifolds:

- **Torus** (circle x circle, closed, no poles): the seeded
  Laplace-Beltrami (``div ∘ raise ∘ grad`` through the kinds)
  converges at 2nd order to the analytic value; the divergence
  theorem holds to rounding — the flux-form ``div`` and the
  sqrt(g)-weighted ``integrate`` share one sqrt(g) derivation, so
  the periodic flux differences telescope *exactly*; curl of grad
  vanishes to rounding (the staggered FD stencils along different
  axes commute algebraically, so the discrete mixed partials cancel
  before the pointwise 1/sqrt(g) scale).
- **Sphere** (periodic lon x bounded lat, polar caps excluded):
  divergence of solid-body rotation vanishes to rounding (sqrt(g)
  is lon-independent, so the lon flux difference of a constant
  angular velocity is exactly zero and the lat flux is identically
  zero); gradient and Laplace-Beltrami of sin(lat) converge at 2nd
  order. The wall-adjacent closures opt into the one-sided
  boundary rows exactly like the C1 bounded-column precedent.
- **Identity chart** (X = (x, y)): every metric-aware entry
  reproduces the flat operators to rounding (the metric derivation
  yields exact ones/zeros, and scaling by 1.0 is exact).

Plus the C2 gates: the traced halo demand of the composed
Laplace-Beltrami is pinned, jit compiles the entry once across
operand values, and the eager operator applications are
device-count invariant (forced-4 suite).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import trace_halo
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.scalars import Variance

RESOLUTIONS = (32, 64, 128)

#: minimum observed convergence order for a 2nd-order scheme
ORDER_FLOOR = 1.7

TWO_PI = 2.0 * jnp.pi

#: torus radii (major, minor)
RING, MINOR = 2.0, 0.7

#: sphere radius and the polar-cap-excluding latitude bound
RADIUS = 1.0
LAT_MAX = float(jnp.deg2rad(80.0))


def observed_orders(errors):
    """Pairwise log2 error ratios of a dyadic refinement chain."""
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


def build_torus(n, device_ids=None):
    """Torus chart grid: two periodic circles, R > r."""
    mu = IntervalMesh(n, (0.0, float(TWO_PI)), name="u")
    mv = IntervalMesh(n, (0.0, float(TWO_PI)), name="v")
    mapping = CoordinateMapping(chart={"X": lambda u, v: (
        (RING + MINOR * jnp.cos(v)) * jnp.cos(u),
        (RING + MINOR * jnp.cos(v)) * jnp.sin(u),
        MINOR * jnp.sin(v))})
    grid = Grid((mu, mv), mapping=mapping, device_ids=device_ids)
    return grid, mu, mv


def build_sphere(n):
    """Lat-lon sphere chart, polar caps excluded (wall BCs)."""
    mlon = IntervalMesh(2 * n, (0.0, float(TWO_PI)), name="lon")
    mlat = IntervalMesh(n, (-LAT_MAX, LAT_MAX), periodic=False,
                        name="lat")
    mapping = CoordinateMapping(chart={"X": lambda lon, lat: (
        RADIUS * jnp.cos(lat) * jnp.cos(lon),
        RADIUS * jnp.cos(lat) * jnp.sin(lon),
        RADIUS * jnp.sin(lat))})
    grid = Grid((mlon, mlat), mapping=mapping)
    # wall-adjacent closures: the BC-free bounded Inner -> Center
    # rows opt into the one-sided boundary patches (the C1
    # bounded-column precedent, boundary_plan.md 2d)
    grid.merge_overrides({
        ("diff", mlat.inner): FiniteDifference(
            order=2, boundary="one_sided"),
        ("interpolate", mlat.inner): LinearInterp(
            boundary="one_sided")})
    return grid, mlon, mlat


def torus_scalar(grid, mu, mv):
    """Build a smooth scalar exercising both chart axes."""
    return grid.create_field(
        mu.center * mv.center,
        init=lambda u, v: jnp.sin(v) + jnp.cos(u))


def torus_laplacian_exact(grid, space):
    """Analytic Laplace-Beltrami of ``sin(v) + cos(u)``."""
    u = grid.evaluation_nodes(space, "u").data
    v = grid.evaluation_nodes(space, "v").data
    ring = RING + MINOR * jnp.cos(v)
    return (-jnp.sin(v) * (RING + 2.0 * MINOR * jnp.cos(v))
            / (MINOR**2 * ring)
            - jnp.cos(u) / ring**2)


# ================================================================
#  Torus: the closed-manifold validation case
# ================================================================
def test_torus_laplace_beltrami_converges_second_order():
    errors = []
    for n in RESOLUTIONS:
        grid, mu, mv = build_torus(n)
        f = torus_scalar(grid, mu, mv)
        lap = grid.dispatch.resolve(
            "laplacian", f.function_space.bare)(f)
        exact = torus_laplacian_exact(grid, lap.function_space)
        errors.append(float(jnp.abs(lap.data - exact).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_torus_gradient_is_the_tagged_covariant_derivative():
    errors = []
    for n in RESOLUTIONS:
        grid, mu, mv = build_torus(n)
        f = torus_scalar(grid, mu, mv)
        g = grid.dispatch.resolve("grad", f.function_space.bare)(f)
        assert g.component_names == ("u", "v")
        for component in g:
            assert component.function_space.variance is (
                Variance.COVARIANT)
        u = grid.evaluation_nodes(g["u"].function_space, "u").data
        v = grid.evaluation_nodes(g["v"].function_space, "v").data
        errors.append(max(
            float(jnp.abs(g["u"].data + jnp.sin(u)).max()),
            float(jnp.abs(g["v"].data - jnp.cos(v)).max())))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_torus_divergence_theorem_holds_to_rounding():
    # the flux-form div and the sqrt_g-weighted integral share one
    # sqrt_g derivation, so the periodic flux differences telescope
    # exactly — rounding is the only residual
    grid, mu, mv = build_torus(64)
    uu = grid.create_field(
        mu.right * mv.center,
        init=lambda u, v: jnp.sin(u + 2.0 * v)
        + 0.3 * jnp.cos(2.0 * u)).with_variance(
            Variance.CONTRAVARIANT)
    vv = grid.create_field(
        mu.center * mv.right,
        init=lambda u, v: jnp.cos(3.0 * u - v)).with_variance(
            Variance.CONTRAVARIANT)
    vec = VectorField({"u": uu, "v": vv})
    div = grid.dispatch.resolve(
        "div", uu.function_space.bare)(vec)
    total = float(div.integrate().data.squeeze())
    scale = float(jnp.abs(div.data).max())
    assert scale > 1.0  # the field genuinely diverges pointwise
    assert abs(total) < 1e-12 * scale


def test_torus_curl_of_grad_vanishes_to_rounding():
    # the same staggered FD stencil bound to different axes
    # commutes algebraically (each addend appears once with either
    # sign), so the discrete mixed partials cancel to rounding and
    # the pointwise 1/sqrt_g scale preserves that
    grid, mu, mv = build_torus(64)
    f = torus_scalar(grid, mu, mv)
    space = f.function_space.bare
    g = grid.dispatch.resolve("grad", space)(f)
    curl = grid.dispatch.resolve("curl", space)(g)
    assert curl.function_space.variance is None
    assert float(jnp.abs(curl.data).max()) < 1e-12


def test_torus_area_integral_is_exact():
    # int sqrt_g du dv = 4 pi^2 R r, exact for the constant field
    grid, mu, mv = build_torus(32)
    one = grid.create_field(
        mu.center * mv.center,
        init=lambda u, v: 1.0 + 0.0 * u + 0.0 * v)
    area = float(one.integrate().data.squeeze())
    assert np.isclose(area, float(4.0 * jnp.pi**2 * RING * MINOR),
                      rtol=1e-14)


# ================================================================
#  Sphere: circle x bounded latitude, polar caps excluded
# ================================================================
def test_sphere_solid_body_rotation_is_divergence_free():
    # u^lon = Omega (constant angular velocity), u^lat = 0: the
    # sqrt_g flux is lon-independent, so the discrete divergence is
    # exactly zero (to rounding), at every resolution
    grid, mlon, mlat = build_sphere(32)
    omega = 0.5
    u_lon = grid.create_field(
        mlon.right * mlat.center,
        init=lambda lon, lat: omega + 0.0 * lon
        + 0.0 * lat).with_variance(Variance.CONTRAVARIANT)
    u_lat = grid.create_field(
        mlon.center * mlat.inner,
        init=lambda lon, lat: 0.0 * lon * lat).with_variance(
            Variance.CONTRAVARIANT)
    vec = VectorField({"lon": u_lon, "lat": u_lat})
    div = grid.dispatch.resolve(
        "div", u_lon.function_space.bare)(vec)
    assert float(jnp.abs(div.data).max()) < 1e-12


def test_sphere_gradient_of_sin_lat_converges():
    errors = []
    for n in RESOLUTIONS:
        grid, mlon, mlat = build_sphere(n)
        f = grid.create_field(
            mlon.center * mlat.center,
            init=lambda lon, lat: jnp.sin(lat) + 0.0 * lon)
        g = grid.dispatch.resolve("grad", f.function_space.bare)(f)
        lat = grid.evaluation_nodes(
            g["lat"].function_space, "lat").data
        errors.append(max(
            float(jnp.abs(g["lat"].data - jnp.cos(lat)).max()),
            float(jnp.abs(g["lon"].data).max())))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_sphere_laplacian_of_sin_lat_converges():
    # sin(lat) is the l=1 zonal spherical harmonic:
    # Laplace-Beltrami is -l(l+1)/a^2 sin(lat) = -2 sin(lat)/a^2
    errors = []
    for n in RESOLUTIONS:
        grid, mlon, mlat = build_sphere(n)
        f = grid.create_field(
            mlon.center * mlat.center,
            init=lambda lon, lat: jnp.sin(lat) + 0.0 * lon)
        lap = grid.dispatch.resolve(
            "laplacian", f.function_space.bare)(f)
        lat = grid.evaluation_nodes(
            lap.function_space, "lat").data
        exact = -2.0 * jnp.sin(lat) / RADIUS**2
        errors.append(float(jnp.abs(lap.data - exact).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


# ================================================================
#  Identity chart: the metric-aware entries mimic the flat ones
# ================================================================
def _mimicry_pair(n=32):
    """One chart grid (X = (x, y)) and one flat twin."""
    def meshes():
        return (IntervalMesh(n, (0.0, float(TWO_PI)), name="x"),
                IntervalMesh(n, (0.0, float(TWO_PI)), name="y"))

    mx, my = meshes()
    chart = Grid((mx, my), mapping=CoordinateMapping(
        chart={"X": lambda x, y: (x, y)}))
    fx, fy = meshes()
    flat = Grid((fx, fy))
    return (chart, mx, my), (flat, fx, fy)


def _mimicry_fields(grid, mx, my):
    scalar = grid.create_field(
        mx.center * my.center,
        init=lambda x, y: jnp.sin(x) * jnp.cos(2.0 * y))
    u = grid.create_field(
        mx.right * my.center, init=lambda x, y: jnp.sin(x + y))
    v = grid.create_field(
        mx.center * my.right,
        init=lambda x, y: jnp.cos(x - 2.0 * y))
    return scalar, u, v


def test_identity_chart_reproduces_the_flat_operators():
    (chart, mx, my), (flat, fx, fy) = _mimicry_pair()
    f, u, v = _mimicry_fields(chart, mx, my)
    f2, u2, v2 = _mimicry_fields(flat, fx, fy)
    space = f.function_space.bare
    space2 = f2.function_space.bare

    g = chart.dispatch.resolve("grad", space)(f)
    g2 = flat.dispatch.resolve("grad", space2)(f2)
    for name, name2 in zip(g.component_names, g2.component_names,
                           strict=True):
        assert float(jnp.abs(g[name].data
                             - g2[name2].data).max()) < 1e-15

    con = VectorField({
        "x": u.with_variance(Variance.CONTRAVARIANT),
        "y": v.with_variance(Variance.CONTRAVARIANT)})
    cov = VectorField({
        "x": u.with_variance(Variance.COVARIANT),
        "y": v.with_variance(Variance.COVARIANT)})
    plain = VectorField({"x": u2, "y": v2})

    d = chart.dispatch.resolve("div", space)(con)
    d2 = flat.dispatch.resolve("div", space2)(plain)
    assert float(jnp.abs(d.data - d2.data).max()) < 1e-15

    c = chart.dispatch.resolve("curl", space)(cov)
    c2 = flat.dispatch.resolve("curl", space2)(plain)
    assert float(jnp.abs(c.data - c2.data).max()) < 1e-15

    lap = chart.dispatch.resolve("laplacian", space)(f)
    lap2 = flat.dispatch.resolve("laplacian", space2)(f2)
    assert float(jnp.abs(lap.data - lap2.data).max()) < 1e-15


def test_identity_chart_raise_index_is_the_identity_map():
    (chart, mx, my), _ = _mimicry_pair()
    _, u, v = _mimicry_fields(chart, mx, my)
    cov = VectorField({
        "x": u.with_variance(Variance.COVARIANT),
        "y": v.with_variance(Variance.COVARIANT)})
    raised = chart.dispatch.resolve(
        "raise_index", cov[0].function_space.bare)(cov)
    for name in cov.component_names:
        assert raised[name].function_space.variance is (
            Variance.CONTRAVARIANT)
        assert float(jnp.abs(raised[name].data
                             - cov[name].data).max()) < 1e-15


# ================================================================
#  Gates: halo trace, jit, device-count invariance
# ================================================================
def test_laplace_beltrami_halo_trace_is_pinned():
    # the deepest sync-free path per axis: one staggered diff
    # composed with one cross-term interpolation hop on the
    # periodic axes — two half-cell moves, one whole cell under
    # the two-sided interval accounting — the raise leg's
    # off-diagonal chains ride between the grad and div
    # differences, and every metric scale is pointwise
    # (halo-neutral); traced through the existing tracer
    # machinery with no special-casing
    grid, mu, mv = build_torus(16)
    space = (mu.center * mv.center).bare

    def tendency(f):
        return grid.dispatch.resolve("laplacian", space)(f)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert spec["u"] == 1
    assert spec["v"] == 1


def test_chart_laplacian_traces_through_jit(compile_counter):
    grid, mu, mv = build_torus(16)
    f1 = torus_scalar(grid, mu, mv)
    f2 = f1 * 2.0
    op = grid.dispatch.resolve("laplacian", f1.function_space.bare)

    @jax.jit
    def step(f):
        return op(f)

    compile_counter.reset()
    l1 = step(f1)
    assert compile_counter.count == 1
    l2 = step(f2)
    assert compile_counter.count == 1  # values swept, one trace
    assert jnp.allclose(l2.data, 2.0 * l1.data)


@pytest.mark.multi_device
def test_chart_operators_are_device_count_invariant():
    def run(device_ids):
        grid, mu, mv = build_torus(32, device_ids=device_ids)
        f = torus_scalar(grid, mu, mv)
        space = f.function_space.bare
        lap = grid.dispatch.resolve("laplacian", space)(f)
        uu = grid.create_field(
            mu.right * mv.center,
            init=lambda u, v: jnp.sin(u + 2.0 * v)).with_variance(
                Variance.CONTRAVARIANT)
        vv = grid.create_field(
            mu.center * mv.right,
            init=lambda u, v: jnp.cos(3.0 * u - v)).with_variance(
                Variance.CONTRAVARIANT)
        div = grid.dispatch.resolve("div", space)(
            VectorField({"u": uu, "v": vv}))
        total = float(div.integrate().data.squeeze())
        return np.asarray(lap.data), np.asarray(div.data), total

    lap_many, div_many, total_many = run(None)
    lap_one, div_one, total_one = run((0,))
    assert np.array_equal(lap_many, lap_one)
    assert np.array_equal(div_many, div_one)
    assert abs(total_many) < 1e-12
    assert abs(total_one) < 1e-12
