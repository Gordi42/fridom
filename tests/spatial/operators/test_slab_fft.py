"""Tests for the distributed slab-decomposed spectral pipeline.

Unmarked tests compare an auto-negotiated grid (all available
devices) against an explicit one-device grid or exercise the plan
kernels on a one-device mesh, so they pass on any device count;
under the forced-devices suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) the auto grid is genuinely sharded
and the slab pipeline runs distributed. ``multi_device`` marked
tests additionally inspect plan resolution, the compiled collectives
and the compile-count stability of the distributed path.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.spatial.operators.slab_fft as slab_mod
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.slab_fft import (
    SlabPlan,
    SlabSolve,
    _internal_coeff,
    _slab_geometry,
    resolve_slab_plan,
    symbol_fits,
)
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.scalars import Scalars


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


def rng_data(shape, seed=0):
    return jnp.asarray(
        np.random.default_rng(seed).standard_normal(shape))


class _StubDecomp:

    """Duck-typed decomposition for geometry-only checks."""

    def __init__(self, layout, count):
        self.default_layout = layout
        self.device_count = count


# ================================================================
#  Plan resolution and fallback conditions
# ================================================================
def test_single_device_grid_resolves_no_plan():
    grid = make_grid((8, 8, 8), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    assert resolve_slab_plan(grid, bare) is None
    # the resolution is memoized per (grid, bare space)
    assert resolve_slab_plan(grid, bare) is None


@pytest.mark.multi_device
def test_plan_geometry_on_a_sharded_grid():
    grid = make_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    plan = resolve_slab_plan(grid, bare)
    assert plan is not None
    assert resolve_slab_plan(grid, bare) is plan  # memoized
    assert plan.domain is bare
    assert plan.layout == grid.decomposition.default_layout
    # internal spectral frame: the half spectrum sits on the local
    # axis z (never the sharded x), x and y carry full spectra
    assert plan.coeff.factor("z").scalars is Scalars.REAL
    assert plan.coeff.factor("x").scalars is Scalars.COMPLEX
    assert plan.coeff.factor("y").scalars is Scalars.COMPLEX
    assert plan.coeff.shape == (16, 16, 9)


@pytest.mark.multi_device
def test_walled_grids_fall_back():
    # bounded z composes trig families: not a plain Fourier
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(16, (0.0, 2.0), name="y")
    mz = fr.spatial.meshes.IntervalMesh(16, (0.0, 3.0),
                                        periodic=False, name="z")
    grid = fr.spatial.Grid((mx, my, mz))
    bare = grid.create_field().function_space.bare
    assert resolve_slab_plan(grid, bare) is None


@pytest.mark.multi_device
def test_indivisible_partner_extents_fall_back():
    # x divides the device count but no other axis does: no
    # transpose partner exists (grid sizes divisible by the device
    # count are the supported case)
    grid = make_grid((16, 18, 18))
    bare = grid.create_field().function_space.bare
    assert dict(grid.decomposition.default_layout.device_axes) == {
        "x": "devices"}
    assert resolve_slab_plan(grid, bare) is None


@pytest.mark.multi_device
def test_single_stage_spaces_fall_back():
    grid = make_grid((16,))
    bare = grid.create_field().function_space.bare
    assert resolve_slab_plan(grid, bare) is None


@pytest.mark.multi_device
def test_unresolvable_transforms_fall_back():
    # an all-Constant space carries no transform signature
    grid = make_grid((16, 16))
    constant = grid.factors[0].constant
    assert resolve_slab_plan(grid, constant) is None


@pytest.mark.multi_device
def test_padded_transforms_fall_back(monkeypatch):
    grid = make_grid((16, 16, 16))
    bare = grid.create_field().function_space.bare
    padded = Fourier(grid, pad=degree(2))
    monkeypatch.setattr(slab_mod, "resolve_transform",
                        lambda _grid, _bare: padded)
    assert slab_mod._build_plan(grid, bare) is None


def test_geometry_rejects_unsuitable_layouts():
    grid = make_grid((16, 16, 16), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    transform = Fourier(grid)
    # a replicated default layout shards nothing
    stub = _StubDecomp(Layout({}), 4)
    assert _slab_geometry(stub, transform, bare) is None
    # the sharded coordinate is not a stage axis
    stub = _StubDecomp(Layout({"q": "devices"}), 4)
    assert _slab_geometry(stub, transform, bare) is None
    # the sharded extent does not divide the device count
    stub = _StubDecomp(Layout({"x": "devices"}), 5)
    assert _slab_geometry(stub, transform, bare) is None
    # no transpose partner divides the device count
    grid_b = make_grid((16, 12, 12), device_ids=(0,))
    bare_b = grid_b.create_field().function_space.bare
    stub = _StubDecomp(Layout({"x": "devices"}), 8)
    assert _slab_geometry(
        stub, Fourier(grid_b), bare_b) is None


def test_geometry_picks_a_local_half_axis():
    grid = make_grid((16, 16, 16), device_ids=(0,))
    bare = grid.create_field().function_space.bare
    stub = _StubDecomp(Layout({"x": "devices"}), 4)
    geometry = _slab_geometry(stub, Fourier(grid), bare)
    assert geometry == ("x", "y", "z", ("x", "y", "z"))
    # 2-D real: no third axis exists, the plan runs fully complex
    grid_2d = make_grid((16, 16), device_ids=(0,))
    bare_2d = grid_2d.create_field().function_space.bare
    geometry = _slab_geometry(stub, Fourier(grid_2d), bare_2d)
    assert geometry == ("x", "y", None, ("x", "y"))
    # complex storage: no Hermitian half axis
    bare_c = bare.replace(
        **{n: bare.factor(n).as_complex() for n in bare.names})
    geometry = _slab_geometry(stub, Fourier(grid), bare_c)
    assert geometry == ("x", "y", None, ("x", "y", "z"))


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
