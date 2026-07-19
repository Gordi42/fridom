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
from fridom.spatial.bc import BC
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.distributed_solve import (
    SlabPlan,
    SlabSolve,
    apply_plan_diagonal,
    build_distributed_plan,
    resolve_distributed_plan,
    resolve_distributed_solve,
    symbol_fits,
)
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import (
    ComposedTransform,
    resolve_transform,
)
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.operators.trig import Cosine
from fridom.spatial.spaces.coefficient import CosineSpace, SineSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
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


def make_walled_grid(shape, device_ids=None,
                     periodic=(True, True, False)):
    # the smoke-script walled column: per-axis periodicity (x, y
    # periodic + z bounded by default), so a bounded axis resolves a
    # trig transform and the product is the mixed Fourier x trig case
    names = ("x", "y", "z")[:len(shape)]
    lengths = (1.0, 2.0, 3.0)[:len(shape)]
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            n, (0.0, ln), periodic=p, name=nm)
        for n, ln, p, nm in zip(shape, lengths, periodic, names,
                                strict=True))
    return fr.spatial.Grid(meshes, device_ids=device_ids)


def _bc_sibling(bare, bc):
    # retag every bounded nodal factor to its BC-tagged sibling (the
    # trig-transform origin of the pressure parity); periodic factors
    # pass through (the _neumann_sibling idiom of nonhydro2.pressure)
    replacements = {
        factor.names[0]: factor.mesh.nodal(factor.node_set, bc=bc)
        for factor in bare.factors
        if isinstance(factor, NodalSpace)
        and not getattr(factor.mesh, "periodic", True)}
    return bare.replace(**replacements) if replacements else bare


def _dirichlet_mid(space, axis):
    # the wall-normal gradient of an even (Neumann) pressure vanishes
    # at the wall: the Dirichlet claim on the staggered faces, keying
    # the divergence legs on the BC-structured rows
    factor = space.factor(axis)
    if (isinstance(factor, NodalSpace)
            and not getattr(factor.mesh, "periodic", True)
            and factor.bc.is_free):
        return space.replace(**{axis: factor.mesh.nodal(
            factor.node_set, bc=BC.DIRICHLET)})
    return space


def walled_laplacian_on(grid, solve_space, dsqr=0.25):
    # the parity-even Div @ Diag @ Grad on the retagged solve space,
    # with the Dirichlet-tagged mid legs (the smoke-script chain)
    grad = Gradient().expand(solve_space, grid)
    axes = solve_space.active_axis_names
    mid = tuple(_dirichlet_mid(s, a) for a, s in
                zip(axes, grad.codomains(solve_space), strict=True))
    div = Divergence().expand(mid, grid)
    diag = Diag({axes[-1]: 1.0 / jnp.asarray(dsqr)}, axes=axes)
    return (div @ diag @ grad).scalar()


def walled_solve(shape, device_ids, periodic, bc, data):
    # a SpectralSolve on the retagged walled solve space, paired with
    # the retagged right-hand side (the operand the slab applies to)
    grid = make_walled_grid(shape, device_ids=device_ids,
                            periodic=periodic)
    rhs = grid.create_field(data=jnp.asarray(data))
    solve_space = _bc_sibling(rhs.function_space.bare, bc)
    solve = SpectralSolve(
        walled_laplacian_on(grid, solve_space), grid, solve_space)
    return solve, rhs.retag(solve_space)


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


@pytest.mark.multi_device
def test_apply_plan_diagonal_matches_the_raw_solve():
    # the field-level wrapper (shared by SlabSolve and the composed
    # transform apply) equals the raw plan.solve on the operand's data,
    # and leaves the operand's sharded axis sharded (layout-preserving)
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    transform = resolve_transform(grid, bare)
    plan = resolve_distributed_plan(transform, grid, bare)
    assert plan is not None
    assert not plan.padded
    diag = laplacian_on(grid, bare, dsqr=1e-4).eigenvalues(
        grid, plan.coeff).data
    out = apply_plan_diagonal(plan, rhs, diag)
    ref = plan.solve(jnp.asarray(rhs.data), diag)
    assert np.array_equal(np.asarray(out.data), np.asarray(ref))
    assert not out.function_space.layout.is_local("x")


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
def test_mismatched_layout_is_a_tier_one_taught_error():
    # a foreign operand layout the slab cannot serve would fall back to
    # the naive replicated composite; that composite shards a transform
    # axis, so the Tier-1 guard rejects it (deliberate, no reroute — the
    # solve is only defined on the layout the slab was negotiated for)
    grid = make_grid((16, 16, 16))
    decomp = grid.decomposition
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    solve = SpectralSolve(laplacian_on(grid, bare), grid, bare)
    assert solve.slab is not None
    moved = rhs.reshard(decomp.layout_for(("x",)))
    assert not solve.slab.applies(moved)
    with pytest.raises(NotImplementedError,
                       match="cannot run on this grid"):
        solve(moved)


# ================================================================
#  Mixed (walled) distributed solve: the staged slab schedule
# ================================================================
@pytest.mark.multi_device
def test_walled_mixed_resolves_to_a_staged_slab_solve():
    # the walled column (x, y periodic + z Neumann) distributes
    # through the joint ComposedTransform plan: the trig axis z is the
    # transpose partner (b), a Fourier axis (y) stays local as the
    # rfft half axis (h), and the internal z factor is a Cosine space
    grid = make_walled_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    solve_space = _bc_sibling(bare, BC.NEUMANN)
    transform = resolve_transform(grid, solve_space)
    assert isinstance(transform, ComposedTransform)
    dist = resolve_distributed_solve(
        walled_laplacian_on(grid, solve_space), transform, grid,
        solve_space, 0.0)
    assert isinstance(dist, SlabSolve)
    plan = dist.plan
    # geometry: x sharded (a), the trig axis z the transpose partner
    # (b), the Fourier axis y the local Hermitian half axis (h)
    assert (plan._a, plan._b, plan._h) == (0, 2, 1)
    assert plan._stages is not None
    assert isinstance(plan.coeff.factor("z"), CosineSpace)


@pytest.mark.multi_device
def test_walled_mixed_slab_matches_a_one_device_reference():
    # the fused slab still solves the mixed walled system, but the naive
    # replicated composite that used to be the reference now shards a
    # transform axis and is a Tier-1 taught error (no reroute), so the
    # reference is an explicit device_ids=(0,) solve instead
    data = rng_data((16, 16, 16), seed=1)
    grid = make_walled_grid((16, 16, 16))
    rhs = grid.create_field(data=data)
    solve_space = _bc_sibling(rhs.function_space.bare, BC.NEUMANN)
    solve = SpectralSolve(
        walled_laplacian_on(grid, solve_space), grid, solve_space)
    assert solve.slab is not None
    operand = rhs.retag(solve_space)
    # the naive composite gathers a sharded transform axis -> rejected
    with pytest.raises(NotImplementedError,
                       match="cannot run on this grid"):
        solve.composite(operand)
    # the one-device reference: every axis local, so the composite the
    # slab reproduces is legal there
    one_grid = make_walled_grid((16, 16, 16), device_ids=(0,))
    one_rhs = one_grid.create_field(data=data)
    one_space = _bc_sibling(one_rhs.function_space.bare, BC.NEUMANN)
    one_solve = SpectralSolve(
        walled_laplacian_on(one_grid, one_space), one_grid, one_space)
    one_operand = one_rhs.retag(one_space)
    assert np.allclose(np.asarray(solve(operand).data),
                       np.asarray(one_solve(one_operand).data),
                       rtol=1e-11, atol=1e-13)


@pytest.mark.multi_device
def test_walled_mixed_solve_transposes_without_gathers():
    grid = make_walled_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    solve_space = _bc_sibling(rhs.function_space.bare, BC.NEUMANN)
    solve = SpectralSolve(
        walled_laplacian_on(grid, solve_space), grid, solve_space)
    assert solve.slab is not None
    operand = rhs.retag(solve_space)
    space = operand.function_space

    def run(storage):
        return solve(ScalarField(grid, space, storage))._data

    text = jax.jit(run).lower(operand._data).compile().as_text()
    # the staged schedule is one all_to_all around the family kernels;
    # nothing gathers the spectral cube
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_walled_mixed_plan_round_trips():
    grid = make_walled_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16), seed=2))
    solve_space = _bc_sibling(rhs.function_space.bare, BC.NEUMANN)
    transform = resolve_transform(grid, solve_space)
    plan = resolve_distributed_plan(transform, grid, solve_space)
    assert plan is not None
    x = jnp.asarray(rhs.retag(solve_space).data)
    # the model-free kernel: forward then backward reproduces the real
    # walled operand (synthesis lands real: the Hermitian stage last)
    back = plan.backward(plan.forward(x))
    assert not jnp.iscomplexobj(back)
    assert float(jnp.abs(back - x).max()) < 1e-13


@pytest.mark.multi_device
def test_warm_eager_walled_solve_adds_zero_compiles(compile_counter):
    # the staged mixed pipeline caches its jit-wrapped shard_map
    # callables per plan, exactly like the all-Fourier path
    grid = make_walled_grid((16, 16, 16))
    data = rng_data((16, 16, 16))

    def solve_once():
        rhs = grid.create_field(data=data)
        solve_space = _bc_sibling(rhs.function_space.bare, BC.NEUMANN)
        solve = SpectralSolve(
            walled_laplacian_on(grid, solve_space), grid, solve_space)
        assert solve.slab is not None
        return solve(rhs.retag(solve_space))

    solve_once()
    solve_once()
    compile_counter.reset()
    solve_once()
    assert compile_counter.count == 0


@pytest.mark.multi_device
def test_fully_walled_solve_stays_real():
    # every axis bounded (Neumann): no Fourier axis, so no Hermitian
    # half stage -- and because the trig kernels are real-to-real, the
    # staged pipeline carries REAL data throughout (no complexify /
    # take-real-part round trip). Lock that in: the forward coefficients
    # of a real field are real.
    data = np.random.default_rng(3).standard_normal((16, 16, 16))
    many, rhs_m = walled_solve(
        (16, 16, 16), None, (False, False, False), BC.NEUMANN, data)
    assert many.slab is not None
    plan = many.slab.plan
    assert plan._stages is not None
    assert plan._h is None
    assert not jnp.iscomplexobj(plan.forward(rhs_m.data))
    # replicated one-device reference (the all-trig composite does not
    # partition cleanly on the multi-device mesh; the 1-device path is
    # the honest replicated baseline)
    one, rhs_o = walled_solve(
        (16, 16, 16), (0,), (False, False, False), BC.NEUMANN, data)
    assert one.slab is None
    assert np.allclose(np.asarray(many(rhs_m).data),
                       np.asarray(one(rhs_o).data),
                       rtol=1e-11, atol=1e-13)


@pytest.mark.multi_device
def test_mixed_sine_solve_matches_one_device():
    # x periodic (Fourier) + y, z Dirichlet (Sine): the internal y, z
    # factors are Sine spaces, and the only Fourier axis is the sharded
    # one, so there is no local Hermitian half axis
    data = np.random.default_rng(4).standard_normal((16, 16, 16))
    many, rhs_m = walled_solve(
        (16, 16, 16), None, (True, False, False), BC.DIRICHLET, data)
    assert many.slab is not None
    plan = many.slab.plan
    assert plan._stages is not None
    assert plan._h is None
    assert isinstance(plan.coeff.factor("z"), SineSpace)
    one, rhs_o = walled_solve(
        (16, 16, 16), (0,), (True, False, False), BC.DIRICHLET, data)
    assert one.slab is None
    assert np.allclose(np.asarray(many(rhs_m).data),
                       np.asarray(one(rhs_o).data),
                       rtol=1e-11, atol=1e-13)


# ================================================================
#  Mixed resolution: the decline conditions
# ================================================================
def test_padded_trig_part_falls_back():
    # a padded trig part is not the plain unpadded kernel the fused
    # region supports; the composed plan declines it (any-padded guard)
    grid = make_walled_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    solve_space = _bc_sibling(bare, BC.NEUMANN)
    padded = ComposedTransform((Fourier(grid, axes=("x", "y")),
                                Cosine(grid, axes="z", pad=degree(2))))
    assert build_distributed_plan(padded, grid, solve_space) is None
    # the planner itself also declines (its own padded-part guard)
    assert padded.distributed_forward_plan(solve_space) is None


def test_chebyshev_column_falls_back():
    # a Chebyshev bounded column resolves a ComposedTransform with a
    # Chebyshev part, outside the per-stage-kernel families: the block
    # -diagonal Chebyshev solve stays on the replicated composite
    grid = fr.spatial.Grid((
        fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x"),
        fr.spatial.meshes.IntervalMesh(16, (0.0, 2.0), name="y"),
        ChebyshevMesh(16, (0.0, 3.0), name="z")))
    space = (grid.factors[0].center * grid.factors[1].center
             * grid.factors[2].nodal(NodeSet.OUTER))
    transform = resolve_transform(grid, space)
    assert isinstance(transform, ComposedTransform)
    assert build_distributed_plan(transform, grid, space) is None
    assert resolve_distributed_plan(transform, grid, space) is None


@pytest.mark.multi_device
def test_indivisible_partners_pad():
    # x (16) shards; the partners (18, 18) are indivisible but paddable,
    # so the joint geometry now picks a padded transpose partner (the
    # trig axis z, keeping the Fourier axis y local as the half axis)
    # instead of declining -- the prime-domain fix (Phase 2)
    grid = make_walled_grid((16, 18, 18))
    bare = grid.create_field().function_space.bare
    solve_space = _bc_sibling(bare, BC.NEUMANN)
    transform = resolve_transform(grid, solve_space)
    geom = transform._joint_geometry(solve_space)
    assert geom is not None
    assert geom[:3] == ("x", "z", "y")
    plan = resolve_distributed_plan(transform, grid, solve_space)
    assert plan is not None
    # the padded balanced all-to-all engages (indivisible partner z)
    assert plan.padded


@pytest.mark.multi_device
def test_unpaddable_partner_falls_back():
    # x (16) shards, but the only partners (5, 5) pad so heavily over
    # four devices that a trailing shard empties -- unpaddable, so the
    # joint geometry declines and keeps the replicated composite
    grid = make_walled_grid((16, 5, 5))
    bare = grid.create_field().function_space.bare
    solve_space = _bc_sibling(bare, BC.NEUMANN)
    transform = resolve_transform(grid, solve_space)
    assert transform._joint_geometry(solve_space) is None
    assert resolve_distributed_plan(
        transform, grid, solve_space) is None


# ================================================================
#  Indivisible split axes: the padded balanced all-to-all (Phase 2)
# ================================================================
def test_divisible_plan_is_not_padded():
    # a divisible / single-device plan needs no padded transpose: it
    # runs the true-frame path byte-for-byte
    grid = make_grid((8, 8, 8), device_ids=(0,))
    assert not one_device_plan(grid).padded


@pytest.mark.multi_device
@pytest.mark.parametrize("n", [18, 33], ids=["n18", "n33-prime"])
def test_indivisible_periodic_solve_matches_one_device(n):
    # a triple-periodic indivisible / prime domain keeps the
    # distributed solve (the padded transpose) instead of replicating;
    # the result matches the one-device reference to machine precision
    data = np.random.default_rng(3).standard_normal((n, n, n))

    def solve_on(device_ids):
        grid = make_grid((n, n, n), device_ids=device_ids)
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
        # indivisible extents engage the padded balanced all-to-all
        assert many.slab.plan.padded
    assert not np.isnan(p_many).any()
    assert not np.isinf(p_many).any()
    assert np.allclose(p_many, p_one, rtol=1e-12, atol=1e-13)


@pytest.mark.multi_device
def test_indivisible_mixed_solve_matches_one_device():
    # a walled column with an indivisible Fourier axis (y=18) and an
    # indivisible trig axis (z=18): the mixed distributed solve pads
    # both the sharded axis (x) and the trig transpose partner (z)
    data = np.random.default_rng(7).standard_normal((16, 18, 18))
    many, rhs_m = walled_solve(
        (16, 18, 18), None, (True, True, False), BC.NEUMANN, data)
    assert many.slab is not None
    assert many.slab.plan.padded
    p_many = np.asarray(many(rhs_m).data)
    one, rhs_o = walled_solve(
        (16, 18, 18), (0,), (True, True, False), BC.NEUMANN, data)
    assert one.slab is None
    p_one = np.asarray(one(rhs_o).data)
    assert not np.isnan(p_many).any()
    assert np.allclose(p_many, p_one, rtol=1e-11, atol=1e-13)


@pytest.mark.multi_device
def test_indivisible_solve_transposes_without_gathers():
    # the prime-domain solve compiles to transposes only -- nothing
    # gathers the spectral cube (the diagnosed replication pathology)
    grid = make_grid((18, 18, 18))
    rhs = grid.create_field(data=rng_data((18, 18, 18)))
    bare = rhs.function_space.bare
    solve = SpectralSolve(
        laplacian_on(grid, bare, dsqr=1e-4), grid, bare)
    assert solve.slab is not None
    assert solve.slab.plan.padded
    space = rhs.function_space

    def run(storage):
        return solve(ScalarField(grid, space, storage))._data

    text = jax.jit(run).lower(rhs._data).compile().as_text()
    # the padded transpose is still an all_to_all; the local pad/slice
    # and the reblock lower shard-local (no cube gather)
    assert "all-to-all" in text
    assert "all-gather" not in text
    assert "all-reduce" not in text


@pytest.mark.multi_device
def test_warm_indivisible_solve_adds_zero_compiles(compile_counter):
    # the padded plan caches its jit-wrapped shard_map solve callable
    # and the even-frame reblock callables, so a warmed eager re-run of
    # the prime-domain solve adds zero compiles
    grid = make_grid((18, 18, 18))
    data = rng_data((18, 18, 18))

    def solve_once():
        rhs = grid.create_field(data=data)
        bare = rhs.function_space.bare
        solve = SpectralSolve(
            laplacian_on(grid, bare, dsqr=1e-4), grid, bare)
        assert solve.slab is not None
        assert solve.slab.plan.padded
        return solve(rhs)

    solve_once()
    solve_once()
    compile_counter.reset()
    solve_once()
    assert compile_counter.count == 0


@pytest.mark.multi_device
def test_indivisible_solve_has_no_pad_lane_leak():
    # the padded transpose zero-fills the split-axis pad lanes and the
    # diagonal is zero-padded to match; the eigenvalue divide must not
    # turn them into NaN/inf. An identity (all-ones) diagonal on a prime
    # domain must round-trip finite and reproduce the input through the
    # padded-even frame.
    grid = make_grid((33, 33, 33))
    rhs = grid.create_field(data=rng_data((33, 33, 33)))
    space = rhs.function_space
    bare = space.bare
    transform = resolve_transform(grid, bare)
    plan = resolve_distributed_plan(transform, grid, bare)
    assert plan is not None
    assert plan.padded
    decomp = grid.decomposition
    even = decomp.unpad_even(rhs.storage, space)
    out = plan.solve(even, jnp.ones(plan.coeff.shape))
    result = rhs.with_storage(decomp.pad_even(out, space))
    r = np.asarray(result.data)
    assert not np.isnan(r).any()
    assert not np.isinf(r).any()
    assert np.allclose(r, np.asarray(rhs.data), rtol=1e-12, atol=1e-13)
