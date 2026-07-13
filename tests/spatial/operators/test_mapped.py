"""Tests for fridom.spatial.operators.mapped."""
import jax.numpy as jnp
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import trace_halo
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import (
    EigenbasisError,
    Identity,
    Operator,
    OperatorSum,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.mapped import (
    MappedDerivative,
    MetricScaled,
)
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.operators.verbs import physical_diff

N = 16
TWO_PI = 2.0 * jnp.pi


def depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, float(TWO_PI)), name="x")


@pytest.fixture
def ms():
    return IntervalMesh(N, (0.0, 1.0), name="sigma")


@pytest.fixture
def grid(mx, ms):
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": depth})
    return Grid((mx, ms), mapping=mapping)


# ================================================================
#  MetricScaled
# ================================================================
def test_metric_scaled_validates_its_arguments():
    with pytest.raises(TypeError, match="Operator"):
        MetricScaled("no-op", numerator="dz_dx")
    with pytest.raises(TypeError, match="strings"):
        MetricScaled(Identity(), numerator=1.0)
    with pytest.raises(TypeError, match="strings"):
        MetricScaled(Identity(), numerator="a", denominator=2)


def test_metric_scaled_delegates_signature_and_requirements(mx):
    fd = FiniteDifference(order=2)
    scaled = MetricScaled(fd["x"], numerator="dz_dx")
    assert scaled.target is fd["x"]
    assert scaled.numerator == "dz_dx"
    assert scaled.denominator is None
    assert scaled.codomain(mx.center) is fd.codomain(mx.center)
    assert (scaled.requirements(mx.center)
            == fd.requirements(mx.center))


def test_metric_scaled_multi_domain_codomain_delegates(mx):
    scaled = MetricScaled(Identity(), numerator="dz_dx")
    result = scaled.codomain(mx.center, mx.right)
    assert result == (mx.center, mx.right)


def test_metric_scaled_has_no_symbol(grid, mx):
    scaled = MetricScaled(FiniteDifference(order=2)["x"],
                          numerator="dz_dx")
    with pytest.raises(EigenbasisError, match="metric"):
        scaled.eigenvalues(grid, mx.fourier(origin=mx.center))


def test_metric_scaled_applies_the_derived_coefficient(grid, mx,
                                                       ms):
    space = mx.center * ms.center
    f = grid.create_field(space,
                          init=lambda x, sigma: 1.0 + 0 * x
                          + 0 * sigma)
    scaled = MetricScaled(Identity(), numerator="dz_dsigma")
    out = scaled(f)
    expected = grid.metric(space, "dz_dsigma")
    assert jnp.allclose(out.data, expected.data)
    quotient = MetricScaled(Identity(), numerator="dz_dsigma",
                            denominator="dz_dsigma")
    assert jnp.allclose(quotient(f).data, 1.0)


# ================================================================
#  MappedDerivative structure and binding
# ================================================================
def test_corrections_are_validated():
    with pytest.raises(TypeError, match="corrections"):
        MappedDerivative({"x": "z"})
    with pytest.raises(TypeError, match="corrections"):
        MappedDerivative({1: ("z", "sigma")})


def test_binding_is_by_coordinate_name():
    op = MappedDerivative({"x": ("z", "sigma")})
    assert op.bound_axis is None
    assert op.corrections == {"x": ("z", "sigma")}
    bound = op["x"]
    assert bound.bound_axis == "x"
    with pytest.raises(TypeError, match="already bound"):
        bound["sigma"]
    with pytest.raises(TypeError, match="coordinate name"):
        op[1]


def test_unexpanded_builder_has_no_signature(mx, ms):
    op = MappedDerivative({"x": ("z", "sigma")})
    with pytest.raises(DispatchError, match="expands"):
        op.codomain(mx.center * ms.center)


def test_axis_inference_needs_an_unambiguous_domain(grid, mx, ms):
    op = MappedDerivative({"x": ("z", "sigma")})
    with pytest.raises(ValueError, match="bind explicitly"):
        op.expand(mx.center * ms.center, grid)


# ================================================================
#  MappedDerivative expansion
# ================================================================
def test_uncorrected_axis_falls_back_to_plain_diff(mx, ms):
    mapping = CoordinateMapping(
        maps={"z": lambda sigma: sigma**2 / 2.0 + sigma / 2.0})
    grid = Grid((mx, ms), mapping=mapping)
    op = grid.dispatch.resolve("physical_diff",
                               mx.center * ms.center)
    expanded = op["x"].expand(mx.center * ms.center, grid)
    assert expanded is grid.dispatch.resolve("diff", mx.center)["x"]


def test_coupled_axis_expands_the_sketch_44_sum(grid, mx, ms):
    space = mx.center * ms.center
    op = grid.dispatch.resolve("physical_diff", space)["x"]
    expanded = op.expand(space, grid)
    assert isinstance(expanded, OperatorSum)
    assert expanded.codomain(space) is (mx.right
                                        * ms.center).bare
    # the correction term carries the metric-name pair
    scaled = expanded.terms[1].target
    assert isinstance(scaled, MetricScaled)
    assert scaled.numerator == "dz_dx"
    assert scaled.denominator == "dz_dsigma"


def test_column_axis_expands_the_scaled_derivative(grid, mx, ms):
    space = mx.center * ms.center
    op = grid.dispatch.resolve("physical_diff", space)["sigma"]
    expanded = op.expand(space, grid)
    assert isinstance(expanded, MetricScaled)
    assert expanded.numerator == "dsigma_dz"
    assert expanded.denominator is None


def test_expansion_infers_the_axis_on_1d_domains(ms):
    mapping = CoordinateMapping(
        maps={"z": lambda sigma: sigma**2 / 2.0 + sigma / 2.0})
    grid = Grid((ms,), mapping=mapping)
    f = grid.create_field(
        ms.center, init=lambda sigma: jnp.sin(TWO_PI * sigma))
    # d/dz = dsigma_dz * d/dsigma, with z = (sigma^2 + sigma) / 2
    df = physical_diff(f)
    s = grid.evaluation_nodes(df.function_space).data
    exact = (TWO_PI * jnp.cos(TWO_PI * s)) / (s + 0.5)
    assert jnp.allclose(df.data, exact, rtol=0.02, atol=0.05)


def test_mismatched_correction_codomain_raises(grid, mx, ms):
    class Stay(Operator):

        """Fake interpolate row that never moves the staggering."""

        def codomain(self, domain):
            return domain

        def __getitem__(self, axis):
            return self

        def __call__(self, f):  # pragma: no cover — never applied
            return f

    class Registry:
        def __init__(self, real):
            self.real = real

        def resolve(self, kind, space):
            if kind == "interpolate":
                return Stay()
            return self.real.resolve(kind, space)

    space = mx.center * ms.center
    op = MappedDerivative({"x": ("z", "sigma")}, axis="x")
    with pytest.raises(SpaceMismatchError, match="correction"):
        op.expand(space, Registry(grid.dispatch))


# ================================================================
#  Halo accounting: the composite through the existing tracer
# ================================================================
def test_halo_trace_walks_the_expanded_composite(grid, mx, ms):
    space = mx.center * ms.center

    def tendency(u):
        return physical_diff["x"](u)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    # main term: one diff along x; correction: diff + interp
    # accumulate along sigma (periodic chain), interp along x
    assert spec["x"] == 1
    assert spec["sigma"] == 2


def test_halo_trace_of_the_column_derivative(grid, mx, ms):
    space = mx.center * ms.center

    def tendency(u):
        return physical_diff["sigma"](u)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert spec["x"] == 0
    assert spec["sigma"] == 1


def test_average_family_expansion_needs_no_interpolation(grid, mx,
                                                         ms):
    # the FV derivative keeps the CellAvg space along both axes, so
    # the correction chain is the bare column derivative
    space = mx.cell_avg * ms.cell_avg
    op = grid.dispatch.resolve("physical_diff", space)["x"]
    expanded = op.expand(space, grid)
    assert isinstance(expanded, OperatorSum)
    assert expanded.codomain(space) is space.bare
    scaled = expanded.terms[1].target
    assert isinstance(scaled, MetricScaled)
    assert scaled.target is grid.dispatch.resolve(
        "diff", ms.cell_avg)["sigma"]


# ================================================================
#  MetricScaled: reciprocal (denominator-only) form (stage C2)
# ================================================================
def test_metric_scaled_needs_at_least_one_name():
    with pytest.raises(ValueError, match="at least one"):
        MetricScaled(Identity())


def test_metric_scaled_denominator_only_is_the_reciprocal(grid, mx,
                                                          ms):
    space = mx.center * ms.center
    f = grid.create_field(space,
                          init=lambda x, sigma: 1.0 + 0 * x
                          + 0 * sigma)
    scaled = MetricScaled(Identity(), denominator="dz_dsigma")
    out = scaled(f)
    assert scaled.numerator is None
    assert scaled.denominator == "dz_dsigma"
    expected = grid.metric(space, "dz_dsigma")
    assert jnp.allclose(out.data * expected.data, 1.0)


# ================================================================
#  Dynamic parameter binding (stage C4)
# ================================================================
def test_with_params_returns_a_transient_bound_builder(grid, mx,
                                                       ms):
    op = grid.dispatch.resolve("physical_diff",
                               mx.center * ms.center)
    assert op.params is None
    h = grid.create_field(mx.center, init=depth)
    bound = op.with_params({"H": h})
    assert bound is not op
    assert bound.params == {"H": h}
    assert bound.corrections == op.corrections
    # axis binding preserves the params and vice versa
    assert bound["x"].params == {"H": h}
    assert op["x"].with_params({"H": h}).bound_axis == "x"
    # empty / None params normalize to the static-defaults builder
    assert op.with_params(None).params is None
    assert op.with_params({}).params is None


def test_with_params_threads_into_the_expanded_coefficients(
        grid, mx, ms):
    space = mx.center * ms.center
    h = grid.create_field(mx.center, init=depth)
    op = grid.dispatch.resolve("physical_diff", space)
    expanded = op.with_params({"H": h})["x"].expand(space, grid)
    scaled = expanded.terms[1].target
    assert isinstance(scaled, MetricScaled)
    assert scaled.params == {"H": h}
    column = op.with_params({"H": h})["sigma"].expand(space, grid)
    assert isinstance(column, MetricScaled)
    assert column.params == {"H": h}


def test_params_bound_derivative_reads_the_current_geometry(
        grid, mx, ms):
    # the params-bound builder on the depth-default grid reproduces
    # the static builder of a grid whose default IS the passed H —
    # the current values drive the coefficients, bitwise
    def other_depth(x):
        return 1.0 + 0.1 * jnp.cos(2.0 * x)

    space = mx.center * ms.center
    u = grid.create_field(
        space,
        init=lambda x, sigma: jnp.sin(TWO_PI * sigma) + 0.0 * x)
    op = grid.dispatch.resolve("physical_diff", space)
    h = grid.create_field(mx.center, init=other_depth)
    dx_dynamic = op.with_params({"H": h})["x"](u)
    dz_dynamic = op.with_params({"H": h})["sigma"](u)

    mx2 = IntervalMesh(N, (0.0, float(TWO_PI)), name="x")
    ms2 = IntervalMesh(N, (0.0, 1.0), name="sigma")
    ref = Grid((mx2, ms2), mapping=CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": other_depth}))
    space2 = mx2.center * ms2.center
    u2 = ref.create_field(
        space2,
        init=lambda x, sigma: jnp.sin(TWO_PI * sigma) + 0.0 * x)
    op2 = ref.dispatch.resolve("physical_diff", space2)
    assert jnp.array_equal(dx_dynamic.data, op2["x"](u2).data)
    assert jnp.array_equal(dz_dynamic.data, op2["sigma"](u2).data)
    # and it genuinely differs from the static-default derivative
    assert not jnp.allclose(dx_dynamic.data, op["x"](u).data)


def test_metric_scaled_params_property_and_application(grid, mx,
                                                       ms):
    space = mx.center * ms.center
    f = grid.create_field(
        space, init=lambda x, sigma: 1.0 + 0 * x + 0 * sigma)
    h2 = grid.create_field(mx.center,
                           init=lambda x: 2.0 * depth(x))
    scaled = MetricScaled(Identity(), numerator="dz_dsigma",
                          params={"H": h2})
    assert scaled.params == {"H": h2}
    assert MetricScaled(Identity(),
                        numerator="dz_dsigma").params is None
    out = scaled(f)
    x = grid.evaluation_nodes(space, "x").data
    expected = jnp.broadcast_to(2.0 * depth(x), out.data.shape)
    assert jnp.allclose(out.data, expected)
