"""Tests for the ``Transform`` ABC surface and the static planner."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import Identity
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.spectral import (
    PhaseShift,
    SpectralDerivative,
)
from fridom.spatial.operators.transform import (
    Transform,
    TransformPlan,
    TransformStage,
    _paddable,
    _sibling_origin,
)
from fridom.spatial.operators.trig import Cosine, Sine
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import NodeSet

TWO_PI = 2.0 * jnp.pi


@pytest.fixture
def grid2d():
    # Pinned to one device: these transform-behaviour tests run the
    # naive forward/backward compute path, which the Tier-1 guard
    # rejects on a sharded transform axis. device_ids=(0,) leaves
    # every axis local so the math is exercised at any device count
    # (the dedicated guard tests build their own multi-device grids).
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), name="y")
    return Grid((mx, my), device_ids=(0,))


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


def test_padded_average_origins_land_on_finer_average_spaces():
    # G7: the padded path now names the refined-mesh average sibling
    # (backward) and rebuilds the coarse origin (forward), for both
    # cell and dual (FaceAvg) families.
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    fine_mesh = mx.refined(degree(2).factor)
    padded = Fourier(grid, pad=degree(2))
    for coarse_origin, fine_origin in (
            (mx.cell_avg, fine_mesh.cell_avg),
            (mx.face_avg, fine_mesh.face_avg)):
        coeff = mx.fourier(origin=coarse_origin)
        assert padded.backward_space(coeff) is fine_origin
        assert padded.codomain(fine_origin) is coeff


def test_sibling_origin_rejects_non_nodal_non_average():
    # the coefficient families (Fourier/trig) are neither nodal nor
    # average origins, so the padded sibling map declines them
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    coeff = mx.fourier(origin=mx.center)
    with pytest.raises(NotImplementedError, match="nodal and average"):
        _sibling_origin(coeff, mx.refined(degree(2).factor),
                        operation="backward")


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


# ================================================================
#  Distributed (multi-device) planner
# ================================================================
def _grid3d(shape=(16, 16, 16), device_ids=None):
    names = ("x", "y", "z")
    lengths = (1.0, 2.0, 3.0)
    meshes = tuple(
        IntervalMesh(n, (0.0, ln), periodic=True, name=nm)
        for n, ln, nm in zip(shape, lengths, names, strict=True))
    return Grid(meshes, device_ids=device_ids)


def test_distributed_plan_is_none_on_one_device():
    grid = _grid3d(device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    assert transform.distributed_forward_plan(bare) is None
    # the ineligible result is memoized (membership test, not .get)
    assert transform.distributed_forward_plan(bare) is None


def test_single_device_forward_plan_carries_no_layout(grid2d, field2d):
    op = Fourier(grid2d)
    plan = op.forward_plan(field2d.function_space)
    assert all(stage.layout is None for stage in plan.stages)


@pytest.mark.multi_device
def test_distributed_forward_plan_geometry_and_coeff_frame():
    grid = _grid3d()
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    # the planner's slab geometry: x sharded (a), y the transpose
    # partner (b), z the local Hermitian half axis (h)
    assert transform._distributed_geometry(bare) == (
        "x", "y", "z", ("x", "y", "z"))
    plan = transform.distributed_forward_plan(bare)
    assert plan is not None
    coeff = plan.codomain.bare
    # internal spectral frame: half spectrum on the local axis z,
    # full complexified spectra on x and y
    assert coeff.shape == (16, 16, 9)
    assert coeff.factor("z").scalars is Scalars.REAL
    assert coeff.factor("x").scalars is Scalars.COMPLEX
    assert coeff.factor("y").scalars is Scalars.COMPLEX


def _bounded_grid3d(shape=(16, 16, 16), device_ids=None):
    names = ("x", "y", "z")
    lengths = (1.0, 2.0, 3.0)
    meshes = tuple(
        IntervalMesh(n, (0.0, ln), periodic=False, name=nm)
        for n, ln, nm in zip(shape, lengths, names, strict=True))
    grid = Grid(meshes, device_ids=device_ids)
    space = None
    for mesh in meshes:
        factor = mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
        space = factor if space is None else space * factor
    return grid, space


@pytest.mark.multi_device
def test_distributed_trig_plan_has_no_half_stage():
    # a real-to-real trig family (all axes Neumann-bounded) is not
    # Hermitian: its distributed plan runs fully complex with no
    # half-spectrum stage (the _distributed_geometry _hermitian guard)
    grid, space = _bounded_grid3d()
    transform = resolve_transform(grid, space)
    assert isinstance(transform, Cosine)
    assert not transform._hermitian
    name_a, name_b, name_h, _ = transform._distributed_geometry(space)
    assert name_h is None
    plan = transform.distributed_forward_plan(space)
    assert plan is not None
    assert not any(stage.half for stage in plan.stages)
    # the sharded axis (a) still transforms last, under the pencil
    assert plan.stages[-1].axis == name_a
    assert name_b != name_a


class _StubDecomp:

    """Duck-typed decomposition for geometry-only checks."""

    def __init__(self, layout, count):
        self.default_layout = layout
        self.device_count = count


class _StubGrid:

    """Duck-typed grid exposing only a decomposition."""

    def __init__(self, decomposition):
        self.decomposition = decomposition


def test_distributed_geometry_rejects_unsuitable_layouts(monkeypatch):
    grid = _grid3d(device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)

    def with_decomp(layout, count):
        monkeypatch.setattr(
            transform, "_grid", _StubGrid(_StubDecomp(layout, count)))

    # a replicated default layout shards nothing
    with_decomp(Layout({}), 4)
    assert transform._distributed_geometry(bare) is None
    # the sharded coordinate is not a stage axis
    with_decomp(Layout({"q": "devices"}), 4)
    assert transform._distributed_geometry(bare) is None
    # the sharded extent does not divide the device count
    with_decomp(Layout({"x": "devices"}), 5)
    assert transform._distributed_geometry(bare) is None
    # no transpose partner divides the device count
    grid_b = _grid3d(shape=(16, 12, 12), device_ids=(0,))
    bare_b = grid_b.create_field().function_space.bare
    transform_b = resolve_transform(grid_b, bare_b)
    monkeypatch.setattr(
        transform_b, "_grid",
        _StubGrid(_StubDecomp(Layout({"x": "devices"}), 8)))
    assert transform_b._distributed_geometry(bare_b) is None


def test_paddable_helper():
    # divisible extents are trivially paddable; an indivisible extent
    # is paddable iff its last (ceil-block) shard keeps >= 1 true slot
    assert _paddable(16, 4)          # divisible
    assert _paddable(18, 4)          # ceil 5, last 3
    assert _paddable(33, 4)          # prime, ceil 9, last 6
    assert not _paddable(16, 5)      # ceil 4, last 0 (heavy)
    assert not _paddable(6, 4)       # ceil 2, last 0 (heavy)
    assert not _paddable(5, 4)       # ceil 2, last -1 (heavy)


def test_distributed_geometry_accepts_paddable_indivisible(
        monkeypatch):
    # an indivisible-but-paddable domain (18 over 4) now yields a slab
    # geometry (the padded balanced all-to-all) instead of declining
    grid = _grid3d(shape=(18, 18, 18), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    monkeypatch.setattr(
        transform, "_grid",
        _StubGrid(_StubDecomp(Layout({"x": "devices"}), 4)))
    geom = transform._distributed_geometry(bare)
    assert geom is not None
    # x sharded (a); no divisible partner, so the first paddable one (y)
    assert geom[:3] == ("x", "y", "z")


def test_distributed_geometry_prefers_divisible_partner(monkeypatch):
    # with both a divisible (z=16) and an indivisible (y=18) partner,
    # the planner picks the divisible one -- the byte-identical fast
    # path -- even though y comes first in grid order
    grid = _grid3d(shape=(16, 18, 16), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    monkeypatch.setattr(
        transform, "_grid",
        _StubGrid(_StubDecomp(Layout({"x": "devices"}), 4)))
    name_a, name_b, _name_h, _ = transform._distributed_geometry(bare)
    assert (name_a, name_b) == ("x", "z")


@pytest.mark.multi_device
def test_distributed_forward_plan_layout_annotation():
    grid = _grid3d()
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    name_a, name_b, name_h, _ = transform._distributed_geometry(bare)
    default_layout = grid.decomposition.default_layout
    axis_name = default_layout.device_axes[0][1]
    plan = transform.distributed_forward_plan(bare)
    # the sharded axis transforms last, under the spectral pencil
    assert plan.stages[-1].axis == name_a
    assert plan.stages[-1].layout == Layout({name_b: axis_name})
    assert plan.codomain.layout == Layout({name_b: axis_name})
    # exactly one half stage, the local Hermitian axis, running first
    halves = [stage for stage in plan.stages if stage.half]
    assert [stage.axis for stage in halves] == [name_h]
    assert plan.stages[0].axis == name_h
    # every earlier stage runs under the operand's nodal layout
    assert all(stage.layout == default_layout
               for stage in plan.stages[:-1])


@pytest.mark.multi_device
def test_distributed_backward_plan_reverses_forward():
    grid = _grid3d()
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    forward = transform.distributed_forward_plan(bare)
    backward = transform.distributed_backward_plan(forward.codomain)
    assert backward is not None
    assert backward.stages == tuple(reversed(forward.stages))
    # backward runs coeff (spectral pencil) -> nodal (default layout)
    assert backward.domain is forward.codomain
    assert backward.codomain.bare is bare
    assert (backward.codomain.layout
            == grid.decomposition.default_layout)


@pytest.mark.multi_device
def test_distributed_plans_are_memoized():
    grid = _grid3d()
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    plan = transform.distributed_forward_plan(bare)
    assert transform.distributed_forward_plan(bare) is plan
    back = transform.distributed_backward_plan(plan.codomain)
    assert transform.distributed_backward_plan(plan.codomain) is back


def test_distributed_backward_plan_none_when_ineligible():
    # a single-device operand has no distributed forward plan, so the
    # backward plan (which mirrors the forward stages) is None too
    grid = _grid3d(device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    assert transform.distributed_backward_plan(
        transform.codomain(bare)) is None


# ================================================================
#  Tier-1 guard: a sharded transform axis is illegal
# ================================================================
def test_device_ids_zero_is_never_rejected():
    # a device_ids=(0,) grid leaves every axis local, so the naive
    # forward runs even on a multi-device host (the guard is exempt)
    grid = _grid3d(device_ids=(0,))
    field = grid.create_field()
    out = Fourier(grid).forward(field)
    assert out.function_space.bare == Fourier(grid).codomain(
        field.function_space.bare)


@pytest.mark.multi_device
def test_sharded_transform_axis_forward_is_rejected():
    # x is sharded AND a transform axis: a *standalone* forward (which
    # materializes a coefficient field the storage contract replicates)
    # has no distributed route, so the Tier-1 guard raises the taught
    # error naming the offending axis. This is the rejected case that
    # remains after phase 3 -- the fused forward->diagonal->backward
    # apply is served (see the apply_diagonal tests below), a standalone
    # forward is not.
    grid = _grid3d()
    assert not grid.decomposition.default_layout.is_local("x")
    field = grid.create_field()
    with pytest.raises(NotImplementedError,
                       match=r"shards the transform axis/axes \('x',\)"):
        Fourier(grid).forward(field)


@pytest.mark.multi_device
def test_rejection_message_points_fourier_at_the_fused_route():
    # the narrowed guard names the fused apply_diagonal route for a
    # plain (unpadded) Fourier transform -- the served family
    grid = _grid3d()
    field = grid.create_field()
    with pytest.raises(NotImplementedError, match="apply_diagonal"):
        Fourier(grid).forward(field)


@pytest.mark.multi_device
def test_rejection_message_points_at_resolve_transform():
    # the narrowed Fourier route hint names resolve_transform, so it is
    # accurate for the walled ComposedTransform too (whose apply_diagonal
    # distributes the periodic axes through the slab pipeline)
    grid = _grid3d()
    field = grid.create_field()
    with pytest.raises(NotImplementedError) as excinfo:
        Fourier(grid).forward(field)
    message = str(excinfo.value)
    assert "resolve_transform" in message
    assert "apply_diagonal" in message


@pytest.mark.multi_device
def test_rejection_message_omits_the_route_for_the_trig_family():
    # a real-to-real trig transform (bounded axes) has no fused
    # forward->diagonal->backward route, so the message does not offer
    # apply_diagonal (the narrowing distinguishes the served family)
    grid, space = _bounded_grid3d()
    transform = resolve_transform(grid, space)
    assert isinstance(transform, Cosine)
    field = grid.create_field(space)
    with pytest.raises(NotImplementedError) as excinfo:
        transform.forward(field)
    assert "apply_diagonal" not in str(excinfo.value)


@pytest.mark.multi_device
def test_sharded_transform_axis_backward_is_rejected():
    # the guard covers the backward seam too (operation name in the
    # message); it fires on the sharded transform axis before the plan
    grid = _grid3d()
    field = grid.create_field()
    with pytest.raises(NotImplementedError,
                       match=r"Fourier\.backward cannot run"):
        Fourier(grid).backward(field)


@pytest.mark.multi_device
def test_replicated_operand_is_not_rejected():
    # a deliberately gathered/replicated operand (a bare space, layout
    # None) passes the guard even on a multi-device grid -- the predicate
    # reads the operand's own layout, not the grid default
    grid = _grid3d()
    replicated = SimpleNamespace(
        function_space=grid.create_field().function_space.bare)
    # no raise (returns None)
    assert Fourier(grid)._reject_sharded_transform(
        replicated, "forward") is None


# ================================================================
#  Tier-2 guard: a sharded non-transform axis is illegal too
# ================================================================
@pytest.mark.multi_device
def test_sharded_non_transform_axis_is_rejected_tier_two():
    # x is sharded but the transform acts on the LOCAL axis z only: the
    # FFT axes are device-local (Tier 1 stays silent), yet a standalone
    # forward materializes a coefficient field the storage contract
    # replicates, so GSPMD would silently all-gather the sharded x axis.
    # That naive Tier-2 path is illegal by design -- the taught error
    # names the sharded non-transform axis.
    grid = _grid3d()
    assert not grid.decomposition.default_layout.is_local("x")
    field = grid.create_field()
    with pytest.raises(
            NotImplementedError,
            match=r"shards the non-transform axis/axes \('x',\)"):
        Fourier(grid, axes="z").forward(field)


@pytest.mark.multi_device
def test_tier_two_message_teaches_the_replicated_escape():
    # the Tier-2 taught error points at the honest replicate-then-compute
    # escapes (SpectralSolve allow_replicated + an explicit reshard)
    grid = _grid3d()
    field = grid.create_field()
    with pytest.raises(NotImplementedError) as excinfo:
        Fourier(grid, axes="z").forward(field)
    message = str(excinfo.value)
    assert "allow_replicated" in message
    assert "reshard" in message


@pytest.mark.multi_device
def test_tier_two_guard_ignores_a_constant_transform_axis():
    # the only transform axis (x) is Constant, so nothing is transformed
    # (an identity forward, no coefficient materialized) -- the guard must
    # not fire even though the layout still maps the sharded x axis
    grid = _grid3d()
    mx, my, mz = grid.factors
    field = grid.create_field(mx.constant * my.center * mz.center)
    assert not field.function_space.layout.is_local("x")
    out = Fourier(grid, axes="x").forward(field)
    assert out.function_space is field.function_space


@pytest.mark.multi_device
def test_tier_two_guard_defers_a_sharded_stage_axis_to_tier_one():
    # a sharded STAGE axis is Tier 1's domain, not Tier 2's: the sibling
    # guard returns None (in forward/backward the Tier-1 guard runs first
    # and raises; this checks the deferral branch directly)
    grid = _grid3d()
    field = grid.create_field()
    assert not field.function_space.layout.is_local("x")
    assert Fourier(grid, axes="x")._reject_replicating_transform(
        field, "forward") is None


@pytest.mark.multi_device
def test_tier_two_replicated_operand_is_not_rejected():
    # the sibling guard reads the operand's own layout: a replicated
    # (layout-free) operand passes even with a sharded grid default
    grid = _grid3d()
    replicated = SimpleNamespace(
        function_space=grid.create_field().function_space.bare)
    assert Fourier(grid, axes="z")._reject_replicating_transform(
        replicated, "forward") is None


@pytest.mark.multi_device
def test_tier_two_explicit_reshard_escape_is_device_count_invariant():
    # the honest escape: reshard the operand to the replicated layout
    # first, and the standalone z-only forward runs (guard satisfied),
    # matching a device_ids=(0,) twin bit-for-bit
    grid = _grid3d()
    field = grid.create_field(
        init=lambda x, y, z: (
            jnp.sin(TWO_PI * x) * jnp.cos(TWO_PI * z)
            + 0.1 * jnp.sin(TWO_PI * y)))
    gathered = field.reshard(Layout({}))
    out = Fourier(grid, axes="z").forward(gathered)

    twin = _grid3d(device_ids=(0,))
    twin_field = twin.create_field(
        init=lambda x, y, z: (
            jnp.sin(TWO_PI * x) * jnp.cos(TWO_PI * z)
            + 0.1 * jnp.sin(TWO_PI * y)))
    twin_out = Fourier(twin, axes="z").forward(twin_field)
    assert float(jnp.abs(
        np.asarray(out.data) - np.asarray(twin_out.data)).max()) == 0.0


# ================================================================
#  Transform.apply_diagonal: the fused forward->diagonal->backward
#  route (phase-3 spectral-derivative consumer)
# ================================================================
def _helmholtz_operator():
    """Build the self-adjoint, endo spectral operator ``I - 0.05 Lap``."""
    lap = (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
           + SpectralDerivative()["y"] @ SpectralDerivative()["y"]
           + SpectralDerivative()["z"] @ SpectralDerivative()["z"])
    return Identity() + (-0.05) * lap


def _peaked(grid):
    """Return a smooth multi-mode field on the grid."""
    return grid.create_field(
        init=lambda x, y, z: jnp.exp(
            -((x - 0.5) ** 2 + (y - 1.0) ** 2 + (z - 1.5) ** 2)))


def test_apply_diagonal_single_device_equals_the_plain_sandwich():
    # on one device apply_diagonal is bit-for-bit the plain
    # backward(symbol(forward)) sandwich (the fallback path)
    grid = _grid3d(device_ids=(0,))
    field = _peaked(grid)
    op = _helmholtz_operator()
    transform = Fourier(grid)
    fused = transform.apply_diagonal(
        field, lambda cb: op.eigenvalues(grid, cb))
    coeff = transform.forward(field)
    symbol = op.eigenvalues(grid, coeff.function_space.bare)
    plain = transform.backward(symbol(coeff))
    assert np.array_equal(np.asarray(fused.data),
                          np.asarray(plain.data))


def test_apply_diagonal_rejects_a_foreign_grid_operand():
    # apply_diagonal is grid-bound like forward/backward
    grid = _grid3d(device_ids=(0,))
    other = _grid3d(device_ids=(0,))
    field = _peaked(other)
    op = _helmholtz_operator()
    with pytest.raises(GridMismatchError, match="grid-bound"):
        Fourier(grid).apply_diagonal(
            field, lambda cb: op.eigenvalues(grid, cb))


@pytest.mark.multi_device
def test_apply_diagonal_routes_through_the_fused_transform():
    # x sharded: the standalone forward would be rejected, but the fused
    # apply_diagonal runs it distributed and matches the single-device
    # fallback to tight rounding, layout-preserving (x stays sharded)
    op = _helmholtz_operator()

    def run(device_ids):
        grid = _grid3d(device_ids=device_ids)
        field = _peaked(grid)
        return grid, Fourier(grid).apply_diagonal(
            field, lambda cb: op.eigenvalues(grid, cb))

    many_grid, many = run(None)
    assert not many_grid.decomposition.default_layout.is_local("x")
    assert not many.function_space.layout.is_local("x")
    _, one = run((0,))
    assert np.allclose(np.asarray(many.data), np.asarray(one.data),
                       rtol=0.0, atol=1e-11)


@pytest.mark.multi_device
def test_apply_diagonal_rejects_a_retagging_symbol_when_sharded():
    # a retagging symbol (PhaseShift: Fourier Center -> Fourier Right)
    # has no layout-preserving distributed form, so the fused route
    # raises the taught error naming the retag
    grid = _grid3d()
    field = _peaked(grid)
    with pytest.raises(NotImplementedError, match="retagging symbol"):
        Fourier(grid).apply_diagonal(
            field,
            lambda cb: PhaseShift(NodeSet.RIGHT)["x"].eigenvalues(
                grid, cb))
