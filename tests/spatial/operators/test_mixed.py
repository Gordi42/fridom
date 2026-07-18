"""Tests for ``ComposedTransform`` and ``resolve_transform`` (C5)."""
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import GridMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import (
    ComposedTransform,
    resolve_transform,
)
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.operators.trig import Cosine, Sine
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def walled():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    return Grid((mx, my, mz), device_ids=(0,)), (mx, my, mz)


def _walled_space(meshes, family):
    mx, my, mz = meshes
    if family is Sine:
        zf = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    else:
        zf = mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    return mx.center * my.center * zf


# ================================================================
#  Mixed resolution: the composed transform
# ================================================================
@pytest.mark.parametrize("family", [Sine, Cosine],
                         ids=["sine", "cosine"])
def test_mixed_product_resolves_to_a_composed_transform(walled,
                                                        family):
    grid, meshes = walled
    space = _walled_space(meshes, family)
    # the registry itself deliberately raises on the mixed product
    with pytest.raises(DispatchError, match="mixed product"):
        grid.dispatch.resolve("transform", space)
    tf = resolve_transform(grid, space)
    assert isinstance(tf, ComposedTransform)
    assert tf.grid is grid
    # deduped per-family parts, Hermitian (Fourier) first
    assert len(tf.parts) == 2
    assert isinstance(tf.parts[0], Fourier)
    assert isinstance(tf.parts[1], family)
    # the parts are the registry's own per-factor instances
    mx, _my, _mz = meshes
    assert tf.parts[0] is grid.dispatch.resolve("transform",
                                                mx.center)
    assert tf.parts[1] is grid.dispatch.resolve(
        "transform", space.factor("z"))


@pytest.mark.parametrize("family", [Sine, Cosine],
                         ids=["sine", "cosine"])
def test_composed_matches_the_manual_composition_bitwise(walled,
                                                         family):
    grid, meshes = walled
    space = _walled_space(meshes, family)
    tf = resolve_transform(grid, space)
    fourier, trig = tf.parts
    f = grid.random.normal(space, seed=5)

    coeff = tf.forward(f)
    manual = trig.forward(fourier.forward(f))
    assert coeff.function_space is manual.function_space
    assert jnp.array_equal(coeff.data, manual.data)  # bitwise

    back = tf.backward(coeff)
    manual_back = fourier.backward(trig.backward(manual))
    assert back.function_space is manual_back.function_space
    assert jnp.array_equal(back.data, manual_back.data)  # bitwise


@pytest.mark.parametrize("family", [Sine, Cosine],
                         ids=["sine", "cosine"])
def test_codomain_and_backward_space_thread_the_parts(walled,
                                                      family):
    grid, meshes = walled
    space = _walled_space(meshes, family)
    tf = resolve_transform(grid, space)
    fourier, trig = tf.parts
    coeff = tf.codomain(space)
    # interned identity against the manual per-family threading
    assert coeff is trig.codomain(fourier.codomain(space))
    assert isinstance(coeff.factor("x"), FourierSpace)
    assert isinstance(coeff.factor("y"), FourierSpace)
    trig_space = SineSpace if family is Sine else CosineSpace
    assert isinstance(coeff.factor("z"), trig_space)
    assert tf.backward_space(coeff) is space.bare


def test_real_field_round_trips_to_a_real_field(walled):
    grid, meshes = walled
    space = _walled_space(meshes, Sine)
    tf = resolve_transform(grid, space)
    f = grid.create_field(
        space,
        init=lambda x, y, z: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(jnp.pi * y) * jnp.sin(2 * jnp.pi * z))
    back = tf.backward(tf.forward(f))
    assert back.function_space is f.function_space
    assert not jnp.iscomplexobj(back.data)
    assert jnp.allclose(back.data, f.data, atol=1e-15)


def test_walled_grids_keep_the_staged_trig_path(walled):
    # the all-Fourier rfftn fast path never swallows the trig axis:
    # the trig part has no fused kernel and stays staged; the
    # Fourier part fuses only its own (periodic) axes
    grid, meshes = walled
    space = _walled_space(meshes, Sine)
    tf = resolve_transform(grid, space)
    f = grid.random.normal(space, seed=8)
    fourier, trig = tf.parts
    data = jnp.asarray(f.data)
    plan = trig.forward_plan(space)
    assert trig._forward_fused_kernel(data, plan) is None
    coeff = tf.forward(f)
    plan_b = trig.backward_plan(coeff.function_space)
    assert trig._backward_fused_kernel(
        jnp.asarray(coeff.data), plan_b) is None
    fused = fourier._forward_fused_kernel(
        data, fourier.forward_plan(space))
    assert fused.shape == (N // 2 + 1, 6, space.shape[2])
    back = tf.backward(coeff)
    assert jnp.allclose(back.data, f.data, atol=1e-13)


def test_metadata_is_preserved_through_the_composition(walled):
    grid, meshes = walled
    space = _walled_space(meshes, Cosine)
    tf = resolve_transform(grid, space)
    f = grid.create_field(space, name="p", units="m^2/s^2")
    coeff = tf.forward(f)
    assert coeff.metadata is f.metadata
    assert tf.backward(coeff).metadata is f.metadata


def test_requirements_declare_transpose(walled):
    grid, meshes = walled
    space = _walled_space(meshes, Sine)
    req = resolve_transform(grid, space).requirements(space)
    assert req.halo == 0
    assert req.layout == "transpose"


# ================================================================
#  Non-mixed resolution: identical to the registry
# ================================================================
def test_homogeneous_products_return_the_registry_instance(walled):
    grid, meshes = walled
    mx, my, mz = meshes
    periodic = mx.center * my.center
    assert resolve_transform(grid, periodic) is (
        grid.dispatch.resolve("transform", periodic))
    bounded = mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    assert resolve_transform(grid, bounded) is (
        grid.dispatch.resolve("transform", bounded))


def test_resolution_is_memoized_per_grid_and_space(walled):
    grid, meshes = walled
    space = _walled_space(meshes, Sine)
    assert resolve_transform(grid, space) is (
        resolve_transform(grid, space))


def test_layouts_are_stripped_before_memoization(walled):
    grid, meshes = walled
    space = _walled_space(meshes, Sine)
    f = grid.create_field(space)
    assert resolve_transform(grid, f.function_space) is (
        resolve_transform(grid, space.bare))


# ================================================================
#  Forward order and factor handling
# ================================================================
def test_forward_order_is_hermitian_first_not_grid_order():
    # bounded z listed first: grid coordinate order alone would put
    # the trig family first, but the forward convention is Fourier
    # (Hermitian half spectrum) first, trig on the complex data after
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    grid = Grid((mz, mx), device_ids=(0,))
    space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET) * mx.center
    tf = resolve_transform(grid, space)
    assert isinstance(tf.parts[0], Fourier)
    assert isinstance(tf.parts[1], Sine)
    f = grid.random.normal(space, seed=6)
    back = tf.backward(tf.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-14)


def test_constant_factors_contribute_no_part(walled):
    grid, meshes = walled
    mx, my, mz = meshes
    space = (mx.constant * my.center
             * mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    tf = resolve_transform(grid, space)
    assert isinstance(tf, ComposedTransform)
    assert len(tf.parts) == 2
    f = grid.random.normal(space, seed=7)
    back = tf.backward(tf.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-14)


# ================================================================
#  Failure modes re-raise the registry's error
# ================================================================
def test_unresolvable_factor_spaces_reraise(walled):
    grid, meshes = walled
    _mx, _my, mz = meshes
    # BC-free bounded factor: no transform row (trig origins are
    # BC-tagged), and the space is not a product to compose over
    with pytest.raises(DispatchError, match="no operator registered"):
        resolve_transform(grid, mz.center)


def test_partially_resolvable_mixed_products_reraise(walled):
    grid, meshes = walled
    mx, _my, mz = meshes
    # the x factor resolves (Fourier) but the BC-free z factor has
    # no row: the per-factor pass propagates its DispatchError
    with pytest.raises(DispatchError, match="no operator registered"):
        resolve_transform(grid, mx.center * mz.center)


def test_all_constant_products_reraise(walled):
    grid, meshes = walled
    mx, my, _mz = meshes
    with pytest.raises(DispatchError, match="no operator registered"):
        resolve_transform(grid, mx.constant * my.constant)


# ================================================================
#  Plumbing-constructor validation
# ================================================================
def test_empty_part_chains_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ComposedTransform(())


def test_parts_on_different_grids_are_rejected(walled):
    grid, _meshes = walled
    other = Grid((IntervalMesh(N, (0.0, 1.0), name="x"),), device_ids=(0,))
    with pytest.raises(GridMismatchError, match="share one grid"):
        ComposedTransform((Fourier(grid, axes="x"), Fourier(other)))


# ================================================================
#  Distributed planning: distributed_forward_plan / _joint_geometry
# ================================================================
def _walled_grid_space(shape, periodic, device_ids=None):
    # a walled grid with its Neumann solve space: periodic axes take
    # the cell centers (Fourier), bounded axes the Neumann centers
    # (Cosine), so the product is the mixed Fourier x trig case
    names = ("x", "y", "z")[:len(shape)]
    lengths = (1.0, 2.0, 3.0)[:len(shape)]
    meshes = tuple(
        IntervalMesh(n, (0.0, ln), periodic=p, name=nm)
        for n, ln, p, nm in zip(shape, lengths, periodic, names,
                                strict=True))
    grid = Grid(meshes, device_ids=device_ids)
    space = None
    for mesh, per in zip(meshes, periodic, strict=True):
        factor = (mesh.center if per
                  else mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN))
        space = factor if space is None else space * factor
    return grid, space


class _StubDecomp:

    """Duck-typed decomposition for geometry-only checks."""

    def __init__(self, layout, count):
        self.default_layout = layout
        self.device_count = count


class _StubGrid:

    """Duck-typed grid exposing only a decomposition."""

    def __init__(self, decomposition):
        self.decomposition = decomposition


@pytest.mark.multi_device
def test_joint_geometry_prefers_the_trig_transpose_partner():
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False))
    tf = resolve_transform(grid, space)
    assert isinstance(tf, ComposedTransform)
    name_a, name_b, name_h, stage_names, part_of = (
        tf._joint_geometry(space))
    # x shards; both y (Fourier) and z (trig) divide the device count,
    # and b prefers the trig axis so a Fourier axis (y) stays local as
    # the rfft half axis h
    assert (name_a, name_b, name_h) == ("x", "z", "y")
    assert stage_names == ("x", "y", "z")
    assert not part_of["z"]._hermitian
    assert part_of["y"]._hermitian
    plan = tf.distributed_forward_plan(space)
    assert [s.axis for s in plan.stages if s.half] == ["y"]
    assert plan.stages[-1].axis == "x"
    assert isinstance(plan.codomain.factor("z"), CosineSpace)


@pytest.mark.multi_device
def test_joint_geometry_has_no_half_when_fourier_axis_is_sharded():
    # a 2-D lon-Fourier / lat-trig product: the only Fourier axis is the
    # sharded one (a), so no Fourier axis stays local and the plan runs
    # with no Hermitian half stage (h = None, fully complex internally)
    grid, space = _walled_grid_space((16, 16), (True, False))
    tf = resolve_transform(grid, space)
    name_a, name_b, name_h, _, _ = tf._joint_geometry(space)
    assert (name_a, name_b, name_h) == ("x", "y", None)
    plan = tf.distributed_forward_plan(space)
    assert not any(s.half for s in plan.stages)


@pytest.mark.multi_device
def test_joint_geometry_has_no_half_on_a_complex_domain():
    # a complex-storage mixed domain carries no real half spectrum, so
    # the Hermitian block is skipped and h = None even though a Fourier
    # axis (y) stays local
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False))
    complexified = space.replace(
        **{name: space.factor(name).as_complex()
           for name in space.names})
    tf = resolve_transform(grid, complexified)
    assert isinstance(tf, ComposedTransform)
    _name_a, _name_b, name_h, _, _ = tf._joint_geometry(complexified)
    assert name_h is None


@pytest.mark.multi_device
def test_distributed_forward_plan_is_memoized():
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False))
    tf = resolve_transform(grid, space)
    plan = tf.distributed_forward_plan(space)
    assert plan is not None
    assert tf.distributed_forward_plan(space) is plan


def test_distributed_forward_plan_is_none_on_one_device():
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False), device_ids=(0,))
    tf = resolve_transform(grid, space)
    assert tf.distributed_forward_plan(space) is None
    # the ineligible result is cached (membership test, not .get)
    assert tf.distributed_forward_plan(space) is None


def test_padded_part_declines_the_joint_plan():
    # a padded trig part is not the plain unpadded kernel the fused
    # region lowers to; the joint planner declines the whole chain
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False))
    padded = ComposedTransform((Fourier(grid, axes=("x", "y")),
                                Cosine(grid, axes="z", pad=degree(2))))
    assert padded.distributed_forward_plan(space) is None


def test_joint_geometry_rejects_unsuitable_layouts(monkeypatch):
    grid, space = _walled_grid_space(
        (16, 16, 16), (True, True, False), device_ids=(0,))
    tf = resolve_transform(grid, space)

    def with_decomp(layout, count):
        # the composed transform reads its decomposition through the
        # shared grid of its first part
        monkeypatch.setattr(
            tf.parts[0], "_grid",
            _StubGrid(_StubDecomp(layout, count)))

    # a replicated default layout shards nothing (len(mapped) != 1)
    with_decomp(Layout({}), 4)
    assert tf._joint_geometry(space) is None
    # the sharded coordinate is not a stage axis
    with_decomp(Layout({"q": "devices"}), 4)
    assert tf._joint_geometry(space) is None
    # the sharded extent does not divide the device count
    with_decomp(Layout({"x": "devices"}), 5)
    assert tf._joint_geometry(space) is None
