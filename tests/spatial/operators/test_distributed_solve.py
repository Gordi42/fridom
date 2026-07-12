"""Tests for the fused distributed spectral solve (kernel + resolution).

The resolution flows through the transform's ``distributed_forward_plan``
(the layout-annotated planner) and drives the fused slab ``shard_map``
kernel that now lives in ``distributed_solve``: same two ``all_to_all``,
no gather, device-count invariant. The kernel tests build a ``SlabPlan``
directly on a one-device mesh (device-count agnostic); the
``SpectralSolve`` integration tests compare an auto-negotiated grid (all
available devices) against an explicit one-device grid. ``multi_device``
marked tests need the forced-4-device suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) to run genuinely sharded.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.distributed_solve import (
    SlabPlan,
    SlabSolve,
    build_distributed_plan,
    resolve_distributed_plan,
    resolve_distributed_solve,
    symbol_fits,
)
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.spaces.tensor_product import TensorProductSpace


def make_grid(shape, device_ids=None, periodic=True):
    names = ("x", "y", "z")[:len(shape)]
    lengths = (1.0, 2.0, 3.0)[:len(shape)]
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            n, (0.0, ln), periodic=periodic, name=nm)
        for n, ln, nm in zip(shape, lengths, names, strict=True))
    return fr.spatial.Grid(meshes, device_ids=device_ids)


def laplacian_on(grid, bare, dsqr=1.0):
    grad = Gradient().expand(bare, grid)
    axes = bare.active_axis_names
    div = Divergence().expand(tuple(grad.codomains(bare)), grid)
    diag = Diag({axes[-1]: 1.0 / jnp.asarray(dsqr)}, axes=axes)
    return (div @ diag @ grad).scalar()


def rng_data(shape, seed=0):
    return jnp.asarray(
        np.random.default_rng(seed).standard_normal(shape))


def _internal_coeff(bare, stage_names, half):
    # the plan's internal coefficient space: the half axis keeps its
    # real origin (Hermitian half spectrum), every other stage factor
    # targets the complexified origin (full spectrum)
    mapping = {}
    for name in stage_names:
        factor = bare.factor(name)
        origin = factor if name == half else factor.as_complex()
        mapping[name] = factor.mesh.fourier(origin=origin)
    factors = tuple(
        mapping.get(factor.names[0], factor)
        if len(factor.names) == 1 else factor
        for factor in bare.factors)
    return TensorProductSpace.of(*factors)


def one_device_plan(grid, half="z"):
    # plumbing construction on the grid's (possibly one-device)
    # mesh: the kernels are device-count agnostic
    bare = grid.create_field().function_space.bare
    decomp = grid.decomposition
    names = bare.names
    coeff = _internal_coeff(bare, names, half)
    return SlabPlan(
        mesh=decomp.device_mesh,
        axis_name=decomp.device_mesh.axis_names[0],
        domain=bare, coeff=coeff, layout=decomp.default_layout,
        a=0, b=1, h=None if half is None else names.index(half),
        fft_axes=tuple(range(len(names))),
        real=True)


# ================================================================
#  Resolution and fallback conditions
# ================================================================
def test_none_on_one_device():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    elliptic = laplacian_on(grid, bare)
    assert resolve_distributed_solve(
        elliptic, transform, grid, bare, 0.0) is None


@pytest.mark.multi_device
def test_hlo_is_the_slab_baseline():
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    transform = resolve_transform(grid, bare)
    elliptic = laplacian_on(grid, bare, dsqr=1e-4)
    dist = resolve_distributed_solve(
        elliptic, transform, grid, bare, 0.0)
    assert dist is not None
    space = rhs.function_space

    def run(storage):
        return dist(ScalarField(grid, space, storage))._data

    text = jax.jit(run).lower(rhs._data).compile().as_text()
    # the golden profile: transposes, nothing gathers the cube
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_matches_one_device_solve():
    data = np.random.default_rng(5).standard_normal((16, 16, 16))
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=jnp.asarray(data))
    bare = rhs.function_space.bare
    transform = resolve_transform(grid, bare)
    dist = resolve_distributed_solve(
        laplacian_on(grid, bare, dsqr=1e-4), transform, grid, bare,
        0.0)
    assert dist is not None
    p_many = np.asarray(dist(rhs).data)
    # replicated one-device reference through the ordinary composite
    one = make_grid((16, 16, 16), device_ids=(0,))
    one_rhs = one.create_field(data=jnp.asarray(data))
    one_bare = one_rhs.function_space.bare
    one_solve = SpectralSolve(
        laplacian_on(one, one_bare, dsqr=1e-4), one, one_bare)
    assert one_solve.slab is None
    assert np.allclose(p_many, np.asarray(one_solve(one_rhs).data),
                       rtol=1e-12, atol=1e-14)


def test_padded_fourier_falls_back():
    # a padded (dealiasing) transform is not the plain unpadded
    # Fourier the fused kernel supports; resolution declines it
    grid = make_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    assert build_distributed_plan(
        Fourier(grid, pad=degree(2)), grid, bare) is None


@pytest.mark.multi_device
def test_two_axis_mesh_falls_back():
    # a genuine 2-D device mesh (pencil) is the extension point; the
    # fused 1-D-slab kernel needs a single-axis mesh
    grid = make_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)

    class _PencilMesh:
        axis_names = ("rows", "cols")

    class _PencilDecomp:
        device_mesh = _PencilMesh()

    class _PencilGrid:
        decomposition = _PencilDecomp()

    assert build_distributed_plan(transform, _PencilGrid(), bare) is None


@pytest.mark.multi_device
def test_plan_is_memoized():
    grid = make_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    plan = resolve_distributed_plan(transform, grid, bare)
    assert plan is not None
    assert resolve_distributed_plan(transform, grid, bare) is plan


@pytest.mark.multi_device
def test_warm_solve_adds_zero_compiles(compile_counter):
    grid = make_grid((16, 16, 16))
    data = rng_data((16, 16, 16))

    def solve_once():
        rhs = grid.create_field(data=data)
        bare = rhs.function_space.bare
        transform = resolve_transform(grid, bare)
        dist = resolve_distributed_solve(
            laplacian_on(grid, bare, dsqr=1e-4), transform, grid,
            bare, 0.0)
        assert dist is not None
        return dist(rhs)

    solve_once()
    solve_once()
    compile_counter.reset()
    solve_once()
    assert compile_counter.count == 0


# ================================================================
#  The kernels match the replicated reference (any device count)
# ================================================================
def test_forward_backward_match_the_fused_reference():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    plan = one_device_plan(grid)
    x = rng_data((8, 8, 8))
    fwd = plan.forward(x)
    # the internal frame is the rfftn with the half axis (z) last
    ref = jnp.fft.rfftn(x, axes=(0, 1, 2), norm="forward")
    assert float(jnp.abs(fwd - ref).max()) < 1e-14
    assert float(jnp.abs(plan.backward(fwd) - x).max()) < 1e-13


def test_fully_complex_pipeline_round_trips():
    # the no-half-axis representation of a real domain (2-D case):
    # complex transforms throughout, real part on synthesis
    grid = make_grid((8, 8), device_ids=(0,))
    plan = one_device_plan(grid, half=None)
    x = rng_data((8, 8))
    fwd = plan.forward(x)
    ref = jnp.fft.fftn(x, axes=(0, 1), norm="forward")
    assert float(jnp.abs(fwd - ref).max()) < 1e-14
    back = plan.backward(fwd)
    assert not jnp.iscomplexobj(back)
    assert float(jnp.abs(back - x).max()) < 1e-13


def test_solve_kernel_applies_the_diagonal():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    plan = one_device_plan(grid)
    x = rng_data((8, 8, 8))
    ones = jnp.ones(plan.coeff.shape)
    assert float(jnp.abs(plan.solve(x, ones) - x).max()) < 1e-13
    # broadcast (size-1) diagonals stay replicated and broadcast
    half = plan.solve(x, jnp.full((1, 1, 1), 0.5))
    assert float(jnp.abs(half - 0.5 * x).max()) < 1e-13


def test_solve_rejects_non_broadcast_diagonals():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    plan = one_device_plan(grid)
    x = rng_data((8, 8, 8))
    with pytest.raises(ValueError, match="broadcast-shaped"):
        plan.solve(x, jnp.ones((2, 1, 1)))


def test_symbol_fits_validates_the_internal_space():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    plan = one_device_plan(grid)
    good = Symbol(plan.coeff, jnp.ones((1, 1, plan.coeff.shape[2])))
    assert symbol_fits(plan, good)
    # wrong space (the replicated codomain, half spectrum on x)
    bare = plan.domain
    standard = Fourier(grid).codomain(bare)
    assert not symbol_fits(
        plan, Symbol(standard, jnp.ones((1, 1, 1))))
    # non-broadcast data shape
    assert not symbol_fits(
        plan, Symbol(plan.coeff, jnp.ones((2, 1, 1))))


def test_slab_solve_wrapper_solves_and_guards():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    plan = one_device_plan(grid)
    lap = laplacian_on(grid, plan.domain)
    inverse = lap.eigenvalues(grid, plan.coeff).inverse(0.0)
    slab = SlabSolve(plan, inverse)
    assert slab.plan is plan
    assert slab.inverse_symbol is inverse
    rhs = grid.create_field(data=rng_data((8, 8, 8)))
    assert slab.applies(rhs)
    reference = SpectralSolve(lap, grid, rhs.function_space)
    out = slab(rhs)
    assert out.function_space is rhs.function_space
    ref = reference(rhs)
    assert float(jnp.abs(out.data - ref.data).max()) < 1e-14
    # a foreign bare space does not apply
    complex_field = grid.create_field(
        plan.domain.replace(x=plan.domain.factor("x").as_complex()))
    assert not slab.applies(complex_field)


@pytest.mark.multi_device
def test_ill_shaped_eigenvalues_fall_back():
    # an operator whose eigenvalues materialize on the internal space
    # but do not broadcast over it (a non-endomorphic / mis-shaped
    # diagonal) must keep the replicated composite (spectral_solve
    # symbol_fits guard)
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    lap = laplacian_on(grid, bare)

    class BadEig:

        def eigenvalues(self, grid, space):
            good = lap.eigenvalues(grid, space)
            # a genuinely non-broadcast diagonal on the internal space
            return Symbol(space, jnp.ones((2, *good.data.shape[1:])))

    solve = SpectralSolve(BadEig(), grid, rhs.function_space)
    assert solve.slab is None


# ================================================================
#  Device-count invariance of the full solve
# ================================================================
def test_solve_is_device_count_invariant():
    data = np.random.default_rng(3).standard_normal((16, 16, 16))

    def solve_on(device_ids):
        grid = make_grid((16, 16, 16), device_ids=device_ids)
        rhs = grid.create_field(data=jnp.asarray(data))
        bare = rhs.function_space.bare
        solve = SpectralSolve(
            laplacian_on(grid, bare, dsqr=1e-4), grid, bare)
        return solve, np.asarray(solve(rhs).data)

    many, p_many = solve_on(None)
    one, p_one = solve_on((0,))
    assert one.slab is None
    if many._grid.decomposition.device_count > 1:
        assert many.slab is not None
    # the distributed op order differs from the replicated one:
    # allclose with a tight tolerance replaces bitwise equality
    # (measured ~7e-16 relative at 16^3 on 4 forced devices)
    assert np.allclose(p_many, p_one, rtol=1e-12, atol=1e-14)


def test_complex_domain_solve_is_device_count_invariant():
    rng = np.random.default_rng(4)
    data = (rng.standard_normal((16, 16))
            + 1j * rng.standard_normal((16, 16)))

    def solve_on(device_ids):
        grid = make_grid((16, 16), device_ids=device_ids)
        centers = grid.create_field().function_space.bare
        bare = centers.replace(
            **{n: centers.factor(n).as_complex()
               for n in centers.names})
        rhs = grid.create_field(bare, data=jnp.asarray(data))
        lap = (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
               + SpectralDerivative()["y"]
               @ SpectralDerivative()["y"])
        solve = SpectralSolve(lap, grid, bare)
        return np.asarray(solve(rhs).data)

    assert np.allclose(solve_on(None), solve_on((0,)),
                       rtol=1e-12, atol=1e-14)


# ================================================================
#  Compiled collectives and compile-count stability
# ================================================================
@pytest.mark.multi_device
def test_solve_compiles_to_transposes_without_gathers():
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    solve = SpectralSolve(
        laplacian_on(grid, bare, dsqr=1e-4), grid, bare)
    assert solve.slab is not None
    space = rhs.function_space

    def run(storage):
        return solve(ScalarField(grid, space, storage))._data

    text = jax.jit(run).lower(rhs._data).compile().as_text()
    # the slab transposes are explicit all-to-alls; nothing may
    # gather the spectral cube (the diagnosed replication pathology)
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_warm_eager_solve_adds_zero_compiles(compile_counter):
    # SpectralSolve is rebuilt per application (the pressure module
    # constructs it at trace time): the plan and its jit-wrapped
    # shard_map callables must be cached so repeated eager solves
    # hit jax's tracing cache
    grid = make_grid((16, 16, 16))
    data = rng_data((16, 16, 16))

    def solve_once():
        rhs = grid.create_field(data=data)
        bare = rhs.function_space.bare
        solve = SpectralSolve(
            laplacian_on(grid, bare, dsqr=1e-4), grid, bare)
        assert solve.slab is not None
        return solve(rhs)

    solve_once()
    solve_once()
    compile_counter.reset()
    solve_once()
    assert compile_counter.count == 0


@pytest.mark.multi_device
def test_mismatched_layouts_fall_back_to_the_composite():
    grid = make_grid((16, 16, 16))
    decomp = grid.decomposition
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    solve = SpectralSolve(laplacian_on(grid, bare), grid, bare)
    assert solve.slab is not None
    moved = rhs.reshard(decomp.layout_for(("x",)))
    assert not solve.slab.applies(moved)
    # the replicated composite serves the foreign layout
    assert np.allclose(np.asarray(solve(moved).data),
                       np.asarray(solve(rhs).data),
                       rtol=1e-12, atol=1e-14)
