"""Tests for the transform-plan-driven distributed solve (Stage 2).

The resolution flows through the transform's ``distributed_forward_plan``
(Stage 1) but reuses the proven slab ``shard_map`` kernel, so the solve
must be identical to ``resolve_slab_plan`` + ``SlabSolve`` — same two
``all_to_all``, no gather, bitwise-equal result. ``multi_device`` marked
tests need the forced-4-device suite
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
    build_distributed_plan,
    resolve_distributed_plan,
    resolve_distributed_solve,
)
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.slab_fft import (
    SlabSolve,
    resolve_slab_plan,
)
from fridom.spatial.operators.spectral_solve import SpectralSolve


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


def test_none_on_one_device():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = resolve_transform(grid, bare)
    elliptic = laplacian_on(grid, bare)
    assert resolve_distributed_solve(
        elliptic, transform, grid, bare, 0.0) is None


@pytest.mark.multi_device
def test_matches_slab_bitwise():
    grid = make_grid((16, 16, 16))
    rhs = grid.create_field(data=rng_data((16, 16, 16)))
    bare = rhs.function_space.bare
    transform = resolve_transform(grid, bare)
    elliptic = laplacian_on(grid, bare, dsqr=1e-4)
    dist = resolve_distributed_solve(
        elliptic, transform, grid, bare, 0.0)
    assert dist is not None
    # the reference: the still-live slab resolution
    slab_plan = resolve_slab_plan(grid, bare)
    inverse = elliptic.eigenvalues(grid, slab_plan.coeff).inverse(0.0)
    slab = SlabSolve(slab_plan, inverse)
    # same kernel + same diagonal -> bitwise-identical result
    assert np.array_equal(
        np.asarray(dist(rhs).data), np.asarray(slab(rhs).data))


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
