"""Tests for fridom.spatial.coordinate_mapping."""
import jax.numpy as jnp
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

N = 16
TWO_PI = 2.0 * jnp.pi


def depth(x):
    """Smooth periodic water depth H(x)."""
    return 1.0 + 0.2 * jnp.sin(x)


def depth_x(x):
    """Return the analytic derivative of ``depth``."""
    return 0.2 * jnp.cos(x)


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, float(TWO_PI)), name="x")


@pytest.fixture
def ms():
    return IntervalMesh(N, (0.0, 1.0), name="sigma")


@pytest.fixture
def mapping():
    return CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": depth})


@pytest.fixture
def grid(mx, ms, mapping):
    return Grid((mx, ms), mapping=mapping)


# ================================================================
#  Declaration: names, forms, and construction errors
# ================================================================
def test_param_and_metric_names(mapping):
    assert mapping.param_names == ("H",)
    assert mapping.metric_names == ("dz_dsigma", "dz_dx",
                                    "dsigma_dz")


def test_supplied_metrics_declare_their_names():
    mapping = CoordinateMapping(
        metrics={"foo": lambda sigma: 1.0 + sigma,
                 "bar": lambda x, H: x * H},
        params={"H": depth})
    assert mapping.metric_names == ("foo", "bar")
    assert mapping.param_names == ("H",)


def test_chart_declares_the_induced_metric_names():
    mapping = CoordinateMapping(
        chart={"X": lambda u, v: (jnp.cos(u), jnp.sin(u), v)})
    assert set(mapping.metric_names) == {
        "g_uu", "g_uv", "g_vu", "g_vv",
        "inv_g_uu", "inv_g_uv", "inv_g_vu", "inv_g_vv",
        "sqrt_g"}


def test_multi_base_map_supplies_jacobians_only():
    # an inline analytic map (no params) with two base coordinates
    # supplies both Jacobian entries but no inverse (no unambiguous
    # column) and seeds no derivative kind
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, x: sigma * (1.0 + 0.2 * x)})
    assert mapping.metric_names == ("dz_dsigma", "dz_dx")
    assert mapping._corrections() == {}


def test_empty_declaration_rejected():
    with pytest.raises(ValueError, match="at least one"):
        CoordinateMapping(params={"H": depth})


def test_non_callable_rejected():
    with pytest.raises(TypeError, match="callables"):
        CoordinateMapping(maps={"z": 1.0})
    with pytest.raises(TypeError, match="callables"):
        CoordinateMapping(maps={"z": lambda s: s},
                          params={"H": 2.0})


def test_map_without_base_coordinate_rejected():
    with pytest.raises(ValueError, match="at least one base"):
        CoordinateMapping(maps={"z": lambda H: H},
                          params={"H": depth})


def test_chart_without_base_coordinate_rejected():
    with pytest.raises(ValueError, match="at least one base"):
        CoordinateMapping(chart={"X": lambda H: (H, H)},
                          params={"H": depth})


def test_duplicate_metric_names_rejected():
    with pytest.raises(ValueError, match="duplicate metric"):
        CoordinateMapping(
            maps={"z": lambda sigma: 2.0 * sigma},
            metrics={"dz_dsigma": lambda sigma: sigma})


def test_two_charts_with_shared_coordinates_collide():
    # per-chart derivation is additive, but one coordinate pair may
    # only be covered by one chart (atlases are out of scope)
    with pytest.raises(ValueError, match="duplicate metric"):
        CoordinateMapping(chart={
            "A": lambda u: (jnp.cos(u), jnp.sin(u)),
            "B": lambda u: (u, u)})


# ================================================================
#  Grid binding
# ================================================================
def test_bind_validates_coordinate_names(mx, ms):
    # an undeclared parameter is an unknown coordinate at bind time
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H})
    with pytest.raises(ValueError, match="params="):
        Grid((mx, ms), mapping=mapping)


def test_bind_rejects_map_key_colliding_with_grid_names(mx, ms):
    mapping = CoordinateMapping(
        maps={"sigma": lambda x: 2.0 * x})
    with pytest.raises(ValueError, match="collides"):
        Grid((mx, ms), mapping=mapping)


def test_bind_validates_param_default_coordinates(mx, ms):
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda y: y})
    with pytest.raises(ValueError, match="parameter 'H'"):
        Grid((mx, ms), mapping=mapping)


def test_mapping_is_grid_bound_once(mx, ms, mapping):
    Grid((mx, ms), mapping=mapping)
    other = IntervalMesh(N, (0.0, 1.0), name="q")
    with pytest.raises(ValueError, match="already attached"):
        Grid((mx, ms, other), mapping=mapping)


def test_unattached_mapping_cannot_derive(mapping, mx, ms):
    with pytest.raises(RuntimeError, match="not attached"):
        mapping.metric(mx.center * ms.center, "dz_dsigma")


def test_identity_equality_and_hash(mapping):
    other = CoordinateMapping(maps={"z": lambda s: s})
    assert mapping == mapping  # noqa: PLR0124 — identity semantics
    assert mapping != other
    assert hash(mapping) == id(mapping)


# ================================================================
#  Analytic-map metrics: values and staggered consistency
# ================================================================
def test_map_jacobian_matches_the_parameter(grid, mx, ms):
    space = mx.center * ms.center
    metric = grid.metric(space, "dz_dsigma")
    x = grid.evaluation_nodes(space, "x").data
    assert jnp.allclose(metric.data, depth(x))


def test_inverse_jacobian_is_the_reciprocal(grid, mx, ms):
    space = mx.center * ms.center
    metric = grid.metric(space, "dsigma_dz")
    x = grid.evaluation_nodes(space, "x").data
    assert jnp.allclose(metric.data, 1.0 / depth(x))


def test_cross_term_chains_the_discrete_parameter_derivative(
        grid, mx, ms):
    space = mx.center * ms.center
    metric = grid.metric(space, "dz_dx")
    x = grid.evaluation_nodes(space, "x").data
    s = grid.evaluation_nodes(space, "sigma").data
    # H_x is the discrete (2nd-order) derivative of the default
    assert jnp.allclose(metric.data, s * depth_x(x), atol=2e-2)
    assert not jnp.allclose(metric.data, s * depth_x(x),
                            atol=1e-12)


@pytest.mark.parametrize("attrs", [
    pytest.param(("center", "center"), id="cell-points"),
    pytest.param(("right", "center"), id="u-points"),
    pytest.param(("center", "right"), id="w-points"),
    pytest.param(("cell_avg", "cell_avg"), id="averages"),
])
def test_staggered_consistency_across_spaces(grid, mx, ms, attrs):
    # one owner: the same metric queried on any staggered space
    # evaluates the same expression at that space's own nodes
    space = (getattr(mx, attrs[0]) * getattr(ms, attrs[1]))
    metric = grid.metric(space, "dz_dsigma")
    x = grid.evaluation_nodes(space, "x").data
    expected = jnp.broadcast_to(depth(x), space.shape)
    assert jnp.allclose(metric.data, expected, atol=5e-3)


def test_metric_is_tagged_with_the_querying_space(grid, mx, ms):
    space = mx.right * ms.center
    metric = grid.metric(space, "dz_dsigma")
    assert metric.function_space.bare is (mx.right
                                          * ms.center).bare
    assert metric.name == "dz_dsigma"


def test_metric_broadcasts_constant_untouched_factors(mx, ms):
    # a map touching sigma only: the x factor of the result is
    # replaced by its ConstantSpace, so it broadcasts exactly
    mapping = CoordinateMapping(
        maps={"z": lambda sigma: sigma**2 / 2.0 + sigma / 2.0})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center
    metric = grid.metric(space, "dz_dsigma")
    bare = metric.function_space.bare
    assert bare is TensorProductSpace.of(mx.constant,
                                         ms.center)
    s = grid.evaluation_nodes(space, "sigma").data
    assert jnp.allclose(metric.data, s + 0.5)
    # exact broadcast under the strict algebra
    f = grid.create_field(space)
    assert (f + metric).function_space.bare is space.bare


def test_metric_on_lone_factor_space(mx, ms):
    mapping = CoordinateMapping(
        maps={"z": lambda sigma: sigma**2 / 2.0 + sigma / 2.0})
    grid = Grid((mx, ms), mapping=mapping)
    metric = grid.metric(ms.center, "dz_dsigma")
    s = grid.evaluation_nodes(ms.center).data
    assert jnp.allclose(metric.data, s + 0.5)


# ================================================================
#  Supplied metrics
# ================================================================
def test_supplied_metric_samples_the_nodes(mx, ms):
    mapping = CoordinateMapping(
        metrics={"foo": lambda sigma: 1.0 + sigma})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.right
    metric = grid.metric(space, "foo")
    s = grid.evaluation_nodes(space, "sigma").data
    assert jnp.allclose(metric.data, 1.0 + s)


def test_supplied_metric_takes_parameters(mx, ms):
    mapping = CoordinateMapping(
        metrics={"weighted": lambda sigma, H: sigma * H},
        params={"H": depth})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center
    metric = grid.metric(space, "weighted")
    x = grid.evaluation_nodes(space, "x").data
    s = grid.evaluation_nodes(space, "sigma").data
    assert jnp.allclose(metric.data, s * depth(x))


# ================================================================
#  Embedding form (CS-D1): charts and induced metrics
# ================================================================
def test_circle_chart_induced_metric_is_exact():
    # 1D-in-2D: a circle of radius R has g_tt = R^2 exactly
    radius = 2.0
    mt = IntervalMesh(N, (0.0, float(TWO_PI)), name="t")
    mapping = CoordinateMapping(chart={
        "X": lambda t: (radius * jnp.cos(t),
                        radius * jnp.sin(t))})
    grid = Grid((mt,), mapping=mapping)
    for space in (mt.center, mt.right):
        assert jnp.allclose(grid.metric(space, "g_tt").data,
                            radius**2)
        assert jnp.allclose(grid.metric(space, "sqrt_g").data,
                            radius)
        assert jnp.allclose(grid.metric(space, "inv_g_tt").data,
                            1.0 / radius**2)


def test_torus_chart_induced_metric_is_exact():
    # 2D-in-3D: the torus metric is diagonal with known entries
    major, minor = 2.0, 0.5
    mu = IntervalMesh(N, (0.0, float(TWO_PI)), name="u")
    mv = IntervalMesh(N, (0.0, float(TWO_PI)), name="v")
    mapping = CoordinateMapping(chart={
        "X": lambda u, v: (
            (major + minor * jnp.cos(v)) * jnp.cos(u),
            (major + minor * jnp.cos(v)) * jnp.sin(u),
            minor * jnp.sin(v))})
    grid = Grid((mu, mv), mapping=mapping)
    space = mu.center * mv.right
    v = grid.evaluation_nodes(space, "v").data
    ring = major + minor * jnp.cos(v)
    assert jnp.allclose(grid.metric(space, "g_uu").data, ring**2)
    assert jnp.allclose(grid.metric(space, "g_vv").data, minor**2)
    assert jnp.allclose(grid.metric(space, "g_uv").data, 0.0,
                        atol=1e-13)
    assert jnp.allclose(grid.metric(space, "g_vu").data, 0.0,
                        atol=1e-13)
    assert jnp.allclose(grid.metric(space, "sqrt_g").data,
                        minor * ring)
    assert jnp.allclose(grid.metric(space, "inv_g_uu").data,
                        1.0 / ring**2)
    assert jnp.allclose(grid.metric(space, "inv_g_vv").data,
                        1.0 / minor**2)


def test_chart_with_parameter_field_chains_discretely():
    # a planar curve r(t) as a parameter: g_tt = r^2 + (dr/dt)^2,
    # the dr/dt entering through the discrete chain rule
    mt = IntervalMesh(64, (0.0, float(TWO_PI)), name="t")

    def radius(t):
        return 1.0 + 0.1 * jnp.sin(t)

    mapping = CoordinateMapping(
        chart={"X": lambda t, R: (R * jnp.cos(t),
                                  R * jnp.sin(t))},
        params={"R": radius})
    grid = Grid((mt,), mapping=mapping)
    metric = grid.metric(mt.center, "g_tt")
    t = grid.evaluation_nodes(mt.center).data
    exact = radius(t)**2 + (0.1 * jnp.cos(t))**2
    assert jnp.allclose(metric.data, exact, atol=1e-3)


def test_chart_must_return_a_tuple():
    mt = IntervalMesh(N, (0.0, float(TWO_PI)), name="t")
    mapping = CoordinateMapping(chart={"X": lambda t: t})
    grid = Grid((mt,), mapping=mapping)
    with pytest.raises(TypeError, match="tuple of ambient"):
        grid.metric(mt.center, "g_tt")


# ================================================================
#  The params= overload (CS-D4)
# ================================================================
def test_explicit_params_replace_the_defaults(grid, mx, ms):
    space = mx.center * ms.center
    x = grid.evaluation_nodes(space, "x").data
    # same values as the default: bitwise identical derivation
    h = grid.create_field(mx.center, init=depth)
    same = grid.metric(space, "dz_dsigma", params={"H": h})
    assert jnp.array_equal(
        same.data, grid.metric(space, "dz_dsigma").data)
    # different values: the supplied field wins
    h2 = grid.create_field(mx.center,
                           init=lambda x: 2.0 + 0.0 * x)
    swapped = grid.metric(space, "dz_dsigma", params={"H": h2})
    assert jnp.allclose(swapped.data, 2.0 + 0.0 * x)


def test_explicit_param_on_staggered_factors(grid, mx, ms):
    # a supplied field on staggered x-points interpolates onto the
    # requested staggering through the registry (one owner)
    space = mx.center * ms.center
    h = grid.create_field(mx.right, init=depth)
    metric = grid.metric(space, "dz_dsigma", params={"H": h})
    x = grid.evaluation_nodes(space, "x").data
    assert jnp.allclose(metric.data, depth(x), atol=5e-3)


def test_explicit_param_with_constant_factor(grid, mx, ms):
    space = mx.center * ms.center
    h = grid.create_field(
        TensorProductSpace.of(mx.center, ms.constant),
        init=depth)
    metric = grid.metric(space, "dz_dsigma", params={"H": h})
    x = grid.evaluation_nodes(space, "x").data
    assert jnp.allclose(metric.data, depth(x))


def test_explicit_param_axis_alignment():
    # a two-coordinate parameter supplied in swapped factor order
    # is realigned to the querying space's mesh order
    mx = IntervalMesh(N, (0.0, float(TWO_PI)), name="x")
    my = IntervalMesh(2 * N, (0.0, float(TWO_PI)), name="y")
    ms = IntervalMesh(4, (0.0, 1.0), name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x, y: depth(x) + jnp.cos(y)})
    grid = Grid((mx, my, ms), mapping=mapping)
    space = mx.center * my.center * ms.center
    swapped = grid.create_field(
        TensorProductSpace.of(my.center, mx.center),
        init=lambda y, x: depth(x) + jnp.cos(y))
    metric = grid.metric(space, "dz_dsigma", params={"H": swapped})
    x = grid.evaluation_nodes(space, "x").data
    y = grid.evaluation_nodes(space, "y").data
    assert jnp.allclose(metric.data,
                        jnp.broadcast_to(depth(x) + jnp.cos(y),
                                         space.shape))


# ================================================================
#  Error paths
# ================================================================
def test_unknown_metric_name(grid, mx, ms):
    with pytest.raises(ValueError, match="unknown metric"):
        grid.metric(mx.center * ms.center, "dz_dy")


def test_unknown_param_key(grid, mx, ms):
    h = grid.create_field(mx.center, init=depth)
    with pytest.raises(ValueError, match="unknown parameters"):
        grid.metric(mx.center * ms.center, "dz_dsigma",
                    params={"eta": h})


def test_space_must_resolve_the_dependent_coordinates(grid, mx,
                                                      ms):
    with pytest.raises(ValueError, match="does not resolve"):
        grid.metric(ms.center, "dz_dsigma")
    with pytest.raises(ValueError, match="constant along"):
        grid.metric(
            TensorProductSpace.of(mx.constant, ms.center),
            "dz_dsigma")


def test_metric_rejects_coefficient_factors(grid, mx, ms):
    space = mx.fourier(origin=mx.center) * ms.center
    with pytest.raises(ValueError, match="coefficient-space"):
        grid.metric(space, "dz_dsigma")


def test_explicit_param_from_another_grid(grid, mx, ms):
    other = Grid((IntervalMesh(N, (0.0, float(TWO_PI)),
                               name="x"),))
    h = other.create_field(other.factors[0].center, init=depth)
    with pytest.raises(ValueError, match="different grid"):
        grid.metric(mx.center * ms.center, "dz_dsigma",
                    params={"H": h})


def test_explicit_param_varying_along_undeclared_coordinate(
        grid, mx, ms):
    h = grid.create_field(mx.center * ms.center,
                          init=lambda x, sigma: depth(x) + sigma)
    with pytest.raises(ValueError, match="varies along"):
        grid.metric(mx.center * ms.center, "dz_dsigma",
                    params={"H": h})


def test_default_param_needs_a_center_family(ms):
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="x")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + x})
    grid = Grid((cheb, ms), mapping=mapping)
    with pytest.raises(NotImplementedError, match="no Center"):
        grid.metric(cheb.outer * ms.center, "dz_dsigma")


def test_two_maps_coupling_one_coordinate_rejected(mx, ms):
    mz = IntervalMesh(N, (0.0, 1.0), name="eta")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H,
              "y": lambda eta, H: eta * H},
        params={"H": depth})
    with pytest.raises(ValueError, match="coupled by two"):
        Grid((mx, ms, mz), mapping=mapping)


# ================================================================
#  The chart-coupling seam (stage C2 seeding)
# ================================================================
def test_chart_coords_reports_the_chart_coupling():
    mapping = CoordinateMapping(chart={
        "X": lambda u, v: (jnp.cos(u), jnp.sin(u), v)})
    assert mapping.chart_coords == ("u", "v")


def test_chart_coords_none_without_a_chart():
    mapping = CoordinateMapping(
        maps={"z": lambda sigma: sigma**2})
    assert mapping.chart_coords is None


def test_grid_chart_coords_mirrors_the_seeding_condition():
    # the public grid twin: the chart coordinate family exactly when
    # the chart couples >= 2 coordinates (the metric-aware seeding
    # condition); None on chartless grids and single-coordinate
    # charts (which seed no metric-aware kinds)
    mu = IntervalMesh(N, (0.0, 1.0), name="u")
    mv = IntervalMesh(N, (0.0, 1.0), name="v")
    chart = Grid((mu, mv), mapping=CoordinateMapping(chart={
        "X": lambda u, v: (jnp.cos(u), jnp.sin(u), v)}))
    assert chart.chart_coords == ("u", "v")

    fu = IntervalMesh(N, (0.0, 1.0), name="u")
    fv = IntervalMesh(N, (0.0, 1.0), name="v")
    assert Grid((fu, fv)).chart_coords is None

    cu = IntervalMesh(N, (0.0, 1.0), name="u")
    cv = IntervalMesh(N, (0.0, 1.0), name="v")
    circle = Grid((cu, cv), mapping=CoordinateMapping(chart={
        "X": lambda u: (jnp.cos(u), jnp.sin(u))}))
    assert circle.chart_coords is None
