"""
Terrain-following metrics (coordinate-systems plan, stage C1).

Sketch 4.4 made real on ``z = sigma * H(x)``: the registered
constant-z derivative kind converges at the operator's order to the
analytic ``d/dx|_z`` of a smooth function evaluated in sigma
coordinates (periodic and bounded columns), the literal
field-coefficient algebra of the sketch reproduces the dispatched
composite exactly, the Jacobian-weighted vertical integral matches
the physical integral, and the jit gates of plan section 5 hold:
sweeping H *values* through the ``params=`` overload compiles once,
and ``jax.grad`` through a metric-dependent scalar with respect to
the H values runs.
"""
import jax
import jax.numpy as jnp
import numpy as np

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.verbs import physical_diff

RESOLUTIONS = (32, 64, 128)

#: minimum observed convergence order for a 2nd-order scheme
ORDER_FLOOR = 1.7

TWO_PI = 2.0 * jnp.pi


def depth(x):
    """Smooth periodic water depth H(x)."""
    return 1.0 + 0.2 * jnp.sin(x)


def depth_x(x):
    """Return the analytic derivative of ``depth``."""
    return 0.2 * jnp.cos(x)


def observed_orders(errors):
    """Pairwise log2 error ratios of a dyadic refinement chain."""
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


def build_grid(n, periodic_sigma):
    """Build a terrain-following grid, ``z = sigma * H(x)``."""
    mx = IntervalMesh(n, (0.0, float(TWO_PI)), name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=periodic_sigma,
                      name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": depth})
    return Grid((mx, ms), mapping=mapping), mx, ms


def field_in_sigma(grid, space):
    """F(x, z) = sin(2 pi z / H(x)) sampled in sigma coordinates."""
    return grid.create_field(
        space,
        init=lambda x, sigma: jnp.sin(TWO_PI * sigma) + 0.0 * x)


def exact_ddx_at_constant_z(grid, space):
    """Return the analytic d/dx|_z of F at the space's nodes."""
    x = grid.evaluation_nodes(space, "x").data
    s = grid.evaluation_nodes(space, "sigma").data
    z = s * depth(x)
    h = depth(x)
    return jnp.cos(TWO_PI * z / h) * (
        -TWO_PI * z * depth_x(x) / h**2)


# ================================================================
#  Sketch 4.4: the constant-z derivative converges at 2nd order
# ================================================================
def test_physical_x_derivative_converges_periodic_column():
    errors = []
    for n in RESOLUTIONS:
        grid, mx, ms = build_grid(n, periodic_sigma=True)
        u = field_in_sigma(grid, mx.center * ms.center)
        du = physical_diff["x"](u)
        exact = exact_ddx_at_constant_z(grid, du.function_space)
        errors.append(float(jnp.abs(du.data - exact).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_physical_x_derivative_converges_bounded_column():
    # a bounded column's correction chain needs the wall closure:
    # the one-sided interpolation override reopens the BC-free
    # Inner -> Center row, and the builder resolves it through the
    # registry (module overrides propagate)
    errors = []
    for n in RESOLUTIONS:
        grid, mx, ms = build_grid(n, periodic_sigma=False)
        grid.merge_overrides({
            ("interpolate", ms.inner):
                LinearInterp(boundary="one_sided")})
        u = field_in_sigma(grid, mx.center * ms.center)
        du = physical_diff["x"](u)
        exact = exact_ddx_at_constant_z(grid, du.function_space)
        errors.append(float(jnp.abs(du.data - exact).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_physical_z_derivative_converges():
    errors = []
    for n in RESOLUTIONS:
        grid, mx, ms = build_grid(n, periodic_sigma=True)
        u = field_in_sigma(grid, mx.center * ms.center)
        du = physical_diff["sigma"](u)  # d/dz along the column
        x = grid.evaluation_nodes(du.function_space, "x").data
        s = grid.evaluation_nodes(du.function_space,
                                  "sigma").data
        exact = jnp.cos(TWO_PI * s) * TWO_PI / depth(x)
        errors.append(float(jnp.abs(du.data - exact).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


def test_sketch_44_literal_algebra_matches_the_dispatched_kind():
    # d/dx|_z = d/dx|_sigma - c * d/dsigma, c a dynamic field leaf
    # from grid.metric — the sketch, staggered onto one signature
    grid, mx, ms = build_grid(64, periodic_sigma=True)
    space = mx.center * ms.center
    fd = FiniteDifference(order=2)
    li = LinearInterp()
    target = mx.right * ms.center
    c = (grid.metric(target, "dz_dx")
         / grid.metric(target, "dz_dsigma"))
    ddx_z = fd["x"] - c * (li["x"] @ li["sigma"] @ fd["sigma"])
    u = field_in_sigma(grid, space)
    manual = ddx_z(u)
    dispatched = physical_diff["x"](u)
    assert manual.function_space is dispatched.function_space
    assert jnp.allclose(manual.data, dispatched.data)


# ================================================================
#  Jacobian-weighted vertical integral
# ================================================================
def test_jacobian_weighted_vertical_integral_converges():
    # int_0^H F dz == int_0^1 F (dz/dsigma) dsigma; with F = z^2
    # the exact column integral is H(x)^3 / 3
    errors = []
    for n in RESOLUTIONS:
        grid, mx, ms = build_grid(n, periodic_sigma=False)
        space = mx.center * ms.center
        u = grid.create_field(
            space,
            init=lambda x, sigma: (sigma * depth(x))**2)
        weighted = u * grid.metric(space, "dz_dsigma")
        column = weighted.integrate("sigma")
        x = grid.evaluation_nodes(column.function_space,
                                  "x").data
        errors.append(float(jnp.abs(
            column.data - depth(x)**3 / 3.0).max()))
    assert np.all(observed_orders(errors) > ORDER_FLOOR)


# ================================================================
#  jit gates (plan section 5)
# ================================================================
def test_metric_param_value_sweep_compiles_once(compile_counter):
    grid, mx, ms = build_grid(16, periodic_sigma=True)
    space = mx.center * ms.center

    @jax.jit
    def coefficient_norm(h):
        metric = grid.metric(space, "dz_dx", params={"H": h})
        return (metric.data ** 2).sum()

    h1 = grid.create_field(mx.center, init=depth)
    h2 = grid.create_field(
        mx.center, init=lambda x: 1.0 + 0.1 * jnp.cos(x))
    compile_counter.reset()
    v1 = coefficient_norm(h1)
    assert compile_counter.count == 1
    v2 = coefficient_norm(h2)
    assert compile_counter.count == 1  # values swept, one trace
    assert not jnp.allclose(v1, v2)


def test_grad_through_metric_params_runs():
    grid, mx, ms = build_grid(16, periodic_sigma=True)
    space = mx.center * ms.center

    def loss(h_values):
        h = grid.create_field(mx.center, data=h_values)
        metric = grid.metric(space, "dz_dx", params={"H": h})
        return (metric.data ** 2).sum()

    h0 = depth(grid.evaluation_nodes(mx.center).data)
    gradient = jax.grad(loss)(h0)
    assert gradient.shape == h0.shape
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.abs(gradient).max()) > 0.0


def test_physical_diff_traces_through_jit(compile_counter):
    grid, mx, ms = build_grid(16, periodic_sigma=True)
    space = mx.center * ms.center

    @jax.jit
    def step(u):
        return physical_diff["x"](u)

    u1 = field_in_sigma(grid, space)
    u2 = u1 * 2.0
    compile_counter.reset()
    d1 = step(u1)
    assert compile_counter.count == 1
    d2 = step(u2)
    assert compile_counter.count == 1
    assert jnp.allclose(d2.data, 2.0 * d1.data)
