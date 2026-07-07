"""Tests for the halo-accounting trace (HaloTracer, trace_halo)."""
import pytest

from fridom.framework2.grid.decomposition.halo import (
    HaloSpec,
    HaloTracer,
    VectorTracer,
    trace_halo,
)
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import (
    BinaryOperator,
    Dispatched,
    OperatorRequirements,
    UnaryOperator,
)
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.movement import Reshard
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def space(mx, my):
    return TensorProductSpace.of(mx.center, my.center)


def widths(spec):
    return dict(spec.widths)


class _StandInGrid:

    """Identity-hashed grid stand-in carrying a decomposition."""

    def __init__(self, decomposition):
        self.decomposition = decomposition


# ================================================================
#  The tracer surface
# ================================================================
def test_data_raises_type_error(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    with pytest.raises(TypeError, match="extra_halo"):
        _ = tracer.data


def test_tracer_grid_exposes_the_registry(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    assert tracer.grid.dispatch is grid.dispatch
    assert tracer.function_space is space
    assert tracer.shape == space.shape
    assert "HaloTracer" in repr(tracer)


def test_default_depth_is_zero(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    assert widths(tracer.depth) == {"x": 0, "y": 0}


def test_integrate_mirrors_the_field_stub(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    with pytest.raises(NotImplementedError, match="integrate"):
        tracer.integrate("x")


# ================================================================
#  Generic interception through the operator base
# ================================================================
def test_diff_returns_the_codomain_tracer(grid, space, mx):
    tracer = HaloTracer(space, grid.dispatch)
    result = tracer.diff("x")
    assert isinstance(result, HaloTracer)
    assert result.function_space.bare.factor("x") is mx.right


def test_trace_of_a_single_diff_is_the_operator_halo(grid, space):
    spec = trace_halo(lambda f: f.diff("x"), (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 0}


def test_unsynced_chains_reproduce_the_per_operator_max(grid, space):
    # iteration-1 contract: the base syncs after every application,
    # so a chain accumulates no depth (max, not sum)
    spec = trace_halo(lambda f: f.diff("x").diff("x"),
                      (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 0}


def test_parallel_terms_max_merge(grid, space):
    def tendency(f):
        # independent branches of one tendency (their spaces differ,
        # so they are returned side by side, not added)
        return f.diff("x"), f.diff("y"), f * f

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_to_conversions_trace_through_the_registry(grid, space, mx):
    def tendency(f):
        target = f.function_space.bare.replace(x=mx.right)
        return f.to(target)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 0}


def test_products_trace_through_the_registry(grid, space):
    spec = trace_halo(lambda f: f * f + f / f - abs(f) + f**2,
                      (space,), grid.dispatch)
    assert widths(spec) == {"x": 0, "y": 0}


def test_mixed_operand_reflected_ops_survive(grid, space):
    field = grid.create_field(init=lambda x, y: x + y)

    def tendency(f):
        return (field + f).diff("x"), (field * f).diff("y")

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_scalar_arithmetic_keeps_the_tracer(grid, space):
    def tendency(f):
        return 2.0 * (-f) + (1.0 - f / 3.0) - (1.0 / f)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert widths(spec) == {"x": 0, "y": 0}


def test_composite_chains_trace_factor_by_factor(grid, space):
    op = Dispatched("diff")["y"] @ Dispatched("diff")["x"]

    def tendency(f):
        return op(f)

    spec = trace_halo(tendency, (space,), grid.dispatch)
    # mixed-axis composite: each factor syncs, per-axis max is exact
    assert widths(spec) == {"x": 1, "y": 1}


# ================================================================
#  Reshard resets depth on the moved axes
# ================================================================
def test_reshard_resets_the_moved_axes(grid, space, mx, my):
    decomp = TensorDecomposition(
        meshes=(mx, my), names=("x", "y"),
        halo=HaloSpec({"x": 1, "y": 1}),
        layouts=(Layout({}), Layout({"x": "devices"})))
    stand_in = _StandInGrid(decomp)
    reshard = Reshard(stand_in, Layout({"x": "devices"}))
    tracer = HaloTracer(space, grid.dispatch,
                        HaloSpec({"x": 2, "y": 1}))
    moved = reshard(tracer)
    assert widths(moved.depth) == {"x": 0, "y": 1}
    assert moved.function_space.layout == Layout({"x": "devices"})


# ================================================================
#  The vector stand-in
# ================================================================
def test_trace_halo_wraps_multiple_spaces(grid, space):
    def tendency(state):
        assert isinstance(state, VectorTracer)
        assert len(state) == 2
        assert "c0" in state
        assert state[0] is state["c0"]
        du = state.map(lambda f: f.diff("x"))
        dv = state.replace(c0=state["c1"]).map(lambda f: f.diff("y"))
        return du + du, dv

    spec = trace_halo(tendency, (space, space), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_vector_tracer_arithmetic_is_componentwise(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    vec = VectorTracer({"u": tracer, "v": tracer})
    assert vec.component_names == ("u", "v")
    assert list(iter(vec)) == [tracer, tracer]
    assert (+vec) is vec
    assert (-vec) is vec
    combined = (2.0 * vec - vec / 3.0 + 1.0)
    assert isinstance(combined, VectorTracer)
    assert combined.component_names == ("u", "v")


def test_vector_tracer_replace_rejects_unknown_names(grid, space):
    vec = VectorTracer((HaloTracer(space, grid.dispatch),))
    with pytest.raises(KeyError):
        vec.replace(missing=HaloTracer(space, grid.dispatch))


def test_trace_halo_needs_state_spaces(grid):
    with pytest.raises(ValueError, match="at least one"):
        trace_halo(lambda state: state, (), grid.dispatch)


# ================================================================
#  Edge branches of the interception hooks
# ================================================================
def test_separable_op_on_a_constant_factor_is_identity(grid, my, mx):
    space = TensorProductSpace.of(mx.constant, my.center)
    tracer = HaloTracer(space, grid.dispatch,
                        HaloSpec({"x": 0, "y": 1}))
    result = FiniteDifference()["x"](tracer)
    # identity along the constant factor: the codomain is unchanged
    # and no per-axis demand is recorded (the post-op sync still
    # resets the returned depth, as for every application)
    assert result.function_space is space
    assert widths(result.depth) == {"x": 0, "y": 0}


def test_whole_space_op_grows_every_bindable_axis(grid, space):
    class WholeSpaceSmoother(UnaryOperator):
        def codomain(self, domain):
            return domain

        def requirements(self, domain):  # noqa: ARG002
            return OperatorRequirements(halo=2)

        def _apply(self, f):  # pragma: no cover — tracer-only test
            return f

    spec = trace_halo(WholeSpaceSmoother(), (space,),
                      grid.dispatch)
    assert widths(spec) == {"x": 2, "y": 2}


def test_nary_op_with_halo_grows_the_codomain_axes(grid, space):
    class WideProduct(BinaryOperator):
        def codomain(self, domain_a, domain_b):  # noqa: ARG002
            return domain_a

        def requirements(self, domain):  # noqa: ARG002
            return OperatorRequirements(halo=1)

        def _apply(self, f, g, *more):  # noqa: ARG002 # pragma: no cover
            return f

    spec = trace_halo(lambda f: WideProduct()(f, f),
                      (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_to_on_the_same_space_returns_self(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    assert tracer.to(space.bare) is tracer


def test_scalar_multiply_keeps_the_tracer(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    assert (tracer * 2.0) is tracer


def test_unsupported_operands_fall_through(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    with pytest.raises(TypeError):
        _ = tracer + object()
    with pytest.raises(TypeError):
        _ = tracer * object()
    with pytest.raises(TypeError):
        _ = object() / tracer
    assert tracer.__pow__("no") is NotImplemented


def test_vector_tracer_components_view(grid, space):
    tracer = HaloTracer(space, grid.dispatch)
    vec = VectorTracer({"u": tracer})
    assert vec.components == {"u": tracer}
