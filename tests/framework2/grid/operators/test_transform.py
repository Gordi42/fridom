"""Tests for the ``Transform`` ABC surface and the static planner."""
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.transform import (
    Transform,
    TransformPlan,
    TransformStage,
)
from fridom.spatial.operators.trig import Sine

TWO_PI = 2.0 * jnp.pi


@pytest.fixture
def grid2d():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), name="y")
    return Grid((mx, my))


@pytest.fixture
def field2d(grid2d):
    return grid2d.create_field(
        init=lambda x, y: jnp.sin(TWO_PI * x) * jnp.cos(TWO_PI * y))


# ================================================================
#  Construction and properties
# ================================================================
def test_transform_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Transform(object())


def test_axes_default_is_all_grid_names(grid2d):
    assert Fourier(grid2d).axes == ("x", "y")


def test_axes_are_a_set_normalized_to_grid_order(grid2d):
    # planner convention: deterministic grid coordinate order
    assert Fourier(grid2d, axes=("y", "x")).axes == ("x", "y")


def test_axes_accept_a_single_string(grid2d):
    assert Fourier(grid2d, axes="y").axes == ("y",)


def test_unknown_axis_raises(grid2d):
    with pytest.raises(ValueError, match="unknown transform axis"):
        Fourier(grid2d, axes=("x", "z"))


def test_duplicate_axes_raise(grid2d):
    with pytest.raises(ValueError, match="duplicate"):
        Fourier(grid2d, axes=("x", "x"))


def test_pad_requires_a_padfactor(grid2d):
    with pytest.raises(TypeError, match="PadFactor"):
        Fourier(grid2d, pad=1.5)


def test_grid_and_pad_properties(grid2d):
    op = Fourier(grid2d, pad=degree(2))
    assert op.grid is grid2d
    assert op.pad == degree(2)
    assert Fourier(grid2d).pad is None


def test_dispatch_kind_is_transform(grid2d):
    assert Fourier(grid2d).dispatch_kind == "transform"


def test_transforms_are_whole_space_not_bindable(grid2d):
    with pytest.raises(TypeError, match="fixed signature"):
        _ = Fourier(grid2d)["x"]


def test_requirements_declare_transpose(grid2d):
    req = Fourier(grid2d).requirements(grid2d.factors[0].center)
    assert req.halo == 0
    assert req.layout == "transpose"
    assert req.collective is False


# ================================================================
#  Grid binding
# ================================================================
def test_transforms_are_grid_bound(field2d):
    other = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                  IntervalMesh(6, (0.0, 2.0), name="y")))
    with pytest.raises(GridMismatchError, match="grid-bound"):
        Fourier(other).forward(field2d)


# ================================================================
#  Registry-uniform application (_apply delegates to forward)
# ================================================================
def test_call_equals_forward(grid2d, field2d):
    op = Fourier(grid2d)
    called = op(field2d)
    forwarded = op.forward(field2d)
    assert called.function_space is forwarded.function_space
    assert jnp.allclose(called.data, forwarded.data)


def test_metadata_is_preserved_by_transforms(grid2d):
    f = grid2d.create_field(
        name="u", units="m/s",
        init=lambda x, y: jnp.sin(TWO_PI * x) * (1 + 0 * y))
    op = Fourier(grid2d)
    coeff = op.forward(f)
    assert coeff.metadata is f.metadata
    assert op.backward(coeff).metadata is f.metadata


# ================================================================
#  The static plan object (section 5.1 planner conventions)
# ================================================================
def test_forward_plan_is_a_static_memoized_object(grid2d, field2d):
    op = Fourier(grid2d)
    plan = op.forward_plan(field2d.function_space)
    assert isinstance(plan, TransformPlan)
    assert plan is op.forward_plan(field2d.function_space)
    assert plan.domain is field2d.function_space.bare
    assert all(isinstance(s, TransformStage) for s in plan.stages)


def test_forward_stages_run_in_grid_order(grid2d, field2d):
    # axes passed reversed: the planner still schedules x first
    op = Fourier(grid2d, axes=("y", "x"))
    plan = op.forward_plan(field2d.function_space)
    assert tuple(s.axis for s in plan.stages) == ("x", "y")
    assert tuple(s.index for s in plan.stages) == (0, 1)


def test_only_the_first_stage_is_hermitian(grid2d, field2d):
    plan = Fourier(grid2d).forward_plan(field2d.function_space)
    assert tuple(s.half for s in plan.stages) == (True, False)


def test_backward_runs_the_hermitian_stage_last(grid2d, field2d):
    op = Fourier(grid2d)
    coeff = op.forward(field2d)
    plan = op.backward_plan(coeff.function_space)
    assert tuple(s.axis for s in plan.stages) == ("y", "x")
    assert tuple(s.half for s in plan.stages) == (False, True)


def test_codomain_and_backward_space_are_bare(grid2d, field2d):
    op = Fourier(grid2d)
    codomain = op.codomain(field2d.function_space)
    assert codomain.layout is None
    assert op.backward_space(codomain) is field2d.function_space.bare


# ================================================================
#  Domain validation
# ================================================================
def test_missing_axis_on_the_operand_raises(grid2d):
    mx = grid2d.factors[0]
    f = grid2d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError, match="not a coordinate"):
        Fourier(grid2d).forward(f)


def test_constant_factors_contribute_no_stage(grid2d):
    mx, my = grid2d.factors
    f = grid2d.create_field(
        mx.constant * my.center,
        init=lambda y: jnp.cos(TWO_PI * y))
    op = Fourier(grid2d)
    coeff = op.forward(f)
    factors = coeff.function_space.bare.factors
    assert factors[0] is mx.constant
    assert factors[1] is my.fourier(origin=my.center)
    plan = op.forward_plan(f.function_space)
    assert tuple(s.axis for s in plan.stages) == ("y",)
    back = op.backward(coeff)
    assert back.function_space.bare is f.function_space.bare
    assert jnp.allclose(back.data, f.data)


def test_backward_rejects_foreign_coefficient_families(grid2d,
                                                       field2d):
    coeff = Fourier(grid2d).forward(field2d)
    with pytest.raises(SpaceMismatchError, match="SineSpace"):
        Sine(grid2d).backward(coeff)


def test_backward_rejects_nodal_operands(grid2d, field2d):
    with pytest.raises(SpaceMismatchError, match="FourierSpace"):
        Fourier(grid2d).backward(field2d)


# ================================================================
#  Padded variants: refined-mesh wiring
# ================================================================
def test_padded_transform_builds_refined_meshes_once():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    op1 = Fourier(grid, pad=degree(2))
    op2 = Fourier(grid, pad=degree(2))
    coeff = Fourier(grid).codomain(mx.center)
    fine1 = op1.backward_space(coeff)
    fine2 = op2.backward_space(coeff)
    # refined() memoizes per (mesh, factor): identical fine spaces
    assert fine1 is fine2
    assert fine1.mesh.refined_from is mx
    assert fine1.mesh.n_cells == 12


def test_padded_average_origins_raise():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg,
                          init=lambda x: jnp.sin(TWO_PI * x))
    coeff = Fourier(grid).forward(f)
    padded = Fourier(grid, pad=degree(2))
    with pytest.raises(NotImplementedError, match="average"):
        padded.backward(coeff)


def test_backward_plan_is_memoized(grid2d, field2d):
    op = Fourier(grid2d)
    coeff = op.forward(field2d)
    plan = op.backward_plan(coeff.function_space)
    assert plan is op.backward_plan(coeff.function_space)


def test_two_half_spectrum_factors_are_rejected(grid2d):
    mx, my = grid2d.factors
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    with pytest.raises(SpaceMismatchError, match="at most one"):
        Fourier(grid2d).backward_space(space)


def test_all_constant_operands_are_left_unchanged(grid2d):
    mx, my = grid2d.factors
    f = grid2d.create_field(mx.constant * my.constant)
    op = Fourier(grid2d)
    coeff = op(f)
    assert coeff.function_space is f.function_space
    plan = op.forward_plan(f.function_space)
    assert plan.stages == ()
