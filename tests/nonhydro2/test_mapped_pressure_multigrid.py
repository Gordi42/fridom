r"""B5 gate evidence for the mapped multigrid pressure preconditioner.

The compile-stability (GB-4) and forced-4-device parity (GB-5) gates of
the geometric-multigrid V-cycle selected through
``preconditioner="multigrid"`` on
:class:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver`
(multigrid pathway plan §B5). Prefix-mirrored shard of
``test_mapped_pressure.py`` (AGENTS oversized-module rule): the B3 knob
wiring / hierarchy-shape / spectral-match tests live in the parent file;
this shard carries the trace-structure and multi-device evidence and
duplicates its own small builders.

GB-4 (compile stability)
------------------------
A given ``(pressure_iterations, multigrid_levels)`` configuration
compiles **once**: warm re-solves (and, in a step loop, re-steps) reuse
the trace across right-hand-side values. The lowered HLO is flat in the
CG iteration count — the outer CG recurrence is a ``lax.scan`` whose body
(with the whole V-cycle inside it) is emitted once, so the program size
does not grow with ``pressure_iterations``. It is **not** flat in the
level count: the cycle is a Python recursion unrolled at trace time
(MG-D8), so more levels emit more program — which is exactly why the
level-count gate is "compiles once", not "identical HLO".

GB-5 (multi-device parity)
--------------------------
The multigrid-preconditioned solve at four forced host devices matches
the single-device result to tight rounding (the cross-shard CG
reductions sum in a different order, so the last bits legitimately
differ — the parity gate of ``test_transfer.py``, not bitwise). One
shard drives a coarse level below the shardability floor so the
replicated fallback (MG-D5) runs under decomposition.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.banded import _resolve_tridiagonal_method
from fridom.spatial.operators.krylov import _computational_mean
from fridom.spatial.operators.multigrid import VerticalLineJacobi

TWO_PI = 2.0 * np.pi
DSQR = 0.25
NZ = 8

#: whether the default jax backend is a GPU (gates the cusparse leg)
_ON_GPU = jax.default_backend() == "gpu"


# ================================================================
#  Self-contained builders (each shard duplicates its helpers)
# ================================================================
def depth(x):
    """Smooth periodic water depth H(x) (steep-ish, ratio 3)."""
    return 1.0 + 0.5 * jnp.sin(x)


def steep_depth(x):
    """Steep periodic water depth (ratio 9) — the anisotropic case."""
    return 1.0 + 0.8 * jnp.sin(x)


def build_grid(nx=16, nz=NZ, *, depth_fn=depth, device_ids=None):
    """Terrain-following 2D grid ``zp = sigma * H(x)`` (x periodic)."""
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(maps={"zp": lambda sigma, H: sigma * H},
                                params={"H": depth_fn})
    return Grid((mx, ms), mapping=mapping, device_ids=device_ids), mx, ms


def build_solver(nx=16, *, iterations, levels, method="auto",
                 depth_fn=depth, device_ids=None):
    """Return a multigrid-preconditioned mapped solver and its grid."""
    grid, mx, ms = build_grid(nx, depth_fn=depth_fn, device_ids=device_ids)
    space = mx.center * ms.center
    solver = MappedPressureSolver(
        grid, space, iterations=iterations,
        weights={"sigma": 1.0 / DSQR},
        preconditioner="multigrid", multigrid_levels=levels,
        multigrid_tridiagonal_method=method)
    return solver, grid, space


def mean_free_rhs(grid, space):
    """Return a smooth, decomposition-invariant mean-free rhs."""
    rhs = grid.create_field(
        space,
        init=lambda x, sigma: jnp.exp(
            -((x - 3.0) ** 2 + (sigma - 0.5) ** 2) * 3.0))
    return rhs - _computational_mean(rhs)


def hlo_line_count(*, iterations, levels):
    """Lower the multigrid solve to HLO and count its lines."""
    solver, grid, space = build_solver(iterations=iterations,
                                       levels=levels)
    rhs = mean_free_rhs(grid, space)
    lowered = jax.jit(
        lambda d: solver.solve(rhs.with_data(d)).data).lower(rhs.data)
    return lowered.as_text().count("\n")


# ================================================================
#  GB-4: the configured solve compiles once (warm re-solve is free)
# ================================================================
def test_multigrid_solve_compiles_once_across_rhs_values(compile_counter):
    # the fixed-count V-cycle preconditioner unrolls at trace time and
    # carries no tolerance branching, so one trace serves every rhs
    # value (the fixed-iteration purity of test_krylov, now with the
    # whole hierarchy inside the CG scan body)
    solver, grid, space = build_solver(iterations=6, levels=3)
    rhs1 = mean_free_rhs(grid, space)
    rhs2 = rhs1 * 3.0

    @jax.jit
    def run(rhs):
        return solver.solve(rhs).data

    warm = run(rhs1)
    compile_counter.reset()
    a = run(rhs1)
    assert compile_counter.count == 0  # cache hit
    b = run(rhs2)
    assert compile_counter.count == 0  # values swept, one trace
    # the operator, the V-cycle and project_mean are all linear, so the
    # solve is linear in the rhs — a swept value scales the solution
    assert jnp.allclose(a, warm)
    assert jnp.allclose(b, 3.0 * warm)


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_multigrid_compiles_once_per_level_config(compile_counter, levels):
    # the level-count gate in its committed form: every hierarchy depth
    # compiles ONCE and a warm re-solve (a re-step) retraces nothing.
    # levels=1 is the smoothing-only degenerate cycle, 2/3 the real
    # recursion (16 -> 8 -> 4 on x, and sigma 8 -> 4 too under the GM-D9
    # full-coarsening default)
    solver, grid, space = build_solver(iterations=6, levels=levels)
    rhs1 = mean_free_rhs(grid, space)
    rhs2 = rhs1 * -2.0

    @jax.jit
    def run(rhs):
        return solver.solve(rhs).data

    run(rhs1)
    compile_counter.reset()
    run(rhs1)
    assert compile_counter.count == 0  # re-solve: cache hit
    run(rhs2)
    assert compile_counter.count == 0  # re-step (new value): no retrace


# ================================================================
#  GB-4: the HLO is flat in the CG iteration count (scan body once)
# ================================================================
def test_multigrid_hlo_is_flat_in_the_cg_iteration_count():
    # the outer CG recurrence is a lax.scan: its body (the whole V-cycle
    # included) is emitted once, so the program size does not grow with
    # the iteration budget however many CG steps run
    lines = {k: hlo_line_count(iterations=k, levels=3)
             for k in (4, 8, 20)}
    assert lines[20] == lines[8]  # 2.5x the budget, zero growth
    # small budgets may lower a few dozen lines differently — jax flips
    # its constants-as-arguments choice on the estimated module size, a
    # representation change, not per-iteration growth (test_krylov)
    assert abs(lines[4] - lines[8]) < 200


def test_multigrid_hlo_grows_with_the_level_count():
    # the counterpart the gate deliberately does NOT flatten: the cycle
    # is a Python recursion unrolled at trace time (MG-D8), so more
    # levels emit more program. This is why the level-count gate above
    # is "compiles once", not HLO-equality — confirm the unroll is real
    lines = {k: hlo_line_count(iterations=6, levels=k)
             for k in (1, 2, 3)}
    assert lines[3] > lines[2] > lines[1]


# ================================================================
#  GB-5: forced-4-device parity with the single-device solve
# ================================================================
def _solve_single_vs_forced(nx, *, levels, iterations=8):
    """Run the multigrid solve on 1 and all devices; return both."""
    def run(device_ids):
        solver, grid, space = build_solver(
            nx, iterations=iterations, levels=levels,
            device_ids=device_ids)
        rhs = mean_free_rhs(grid, space)
        return np.asarray(solver.solve(rhs).data)

    ids = tuple(range(jax.device_count()))
    return run((0,)), run(ids)


@pytest.mark.multi_device
@pytest.mark.parametrize(
    ("nx", "levels"),
    [pytest.param(16, 3, id="aligned-x16-3lvl"),
     pytest.param(12, 3, id="replicated-x12-coarse6")])
def test_forced4_multigrid_solve_matches_single_device(nx, levels):
    # If a halo/pad slot leaked into a level's smoother, transfer or the
    # CG inner products, the sharded solve would diverge from the
    # single-device one. The x=12 shard coarsens 12 -> 6 and 6 does not
    # divide four devices, so its coarse level lives replicated on the
    # same device mesh (MG-D5) — exercised here under decomposition.
    one, many = _solve_single_vs_forced(nx, levels=levels)
    scale = np.max(np.abs(one))
    assert np.max(np.abs(one - many)) / scale < 1e-8


# ================================================================
#  Tridiagonal-kernel knob threading (B4 plumbing)
# ================================================================
def test_multigrid_tridiagonal_method_is_stored_and_forwarded():
    # the knob is stored and reaches every level's line smoother
    solver, _, _ = build_solver(iterations=3, levels=2, method="pcr")
    assert solver._multigrid_tridiagonal_method == "pcr"
    vcycle = solver._build_vcycle({})
    assert vcycle.levels
    assert all(level.smoother.method == "pcr"
               for level in vcycle.levels)


def test_multigrid_tridiagonal_method_defaults_to_auto():
    solver, _, _ = build_solver(iterations=3, levels=2)
    assert solver._multigrid_tridiagonal_method == "auto"


def test_multigrid_levels_defaults_to_floor_limited_depth():
    # the None default (omitted) stores None and coarsens to the
    # four-cell floor: nx=16 halves 16 -> 8 -> 4, a three-level V-cycle
    grid, mx, ms = build_grid(nx=16)
    space = mx.center * ms.center
    solver = MappedPressureSolver(
        grid, space, iterations=3, weights={"sigma": 1.0 / DSQR},
        preconditioner="multigrid")
    assert solver._multigrid_levels is None
    vcycle = solver._build_vcycle({})
    assert len(vcycle.levels) == 3


def test_multigrid_levels_int_still_caps_the_depth():
    # an explicit int truncates below the floor depth: cap 2 on the same
    # nx=16 grid stops at two levels (16 -> 8), not the floor's three
    solver, _, _ = build_solver(nx=16, iterations=3, levels=2)
    assert solver._multigrid_levels == 2
    vcycle = solver._build_vcycle({})
    assert len(vcycle.levels) == 2


def test_mapped_solver_rejects_unknown_tridiagonal_method():
    grid, mx, ms = build_grid()
    space = mx.center * ms.center
    with pytest.raises(ValueError,
                       match="tridiagonal method must be one of"):
        MappedPressureSolver(
            grid, space, iterations=3, weights={"sigma": 1.0 / DSQR},
            preconditioner="multigrid",
            multigrid_tridiagonal_method="thomas")


# ================================================================
#  GM-D9: full 3-D coarsening is the mapped-solver default
# ================================================================
def build_grid_3d(nx, ny, nz, *, depth_fn=None):
    """Build a 3-D terrain grid ``zp = sigma H(x, y)`` (x, y periodic)."""
    depth_fn = depth_fn or (
        lambda x, y: 1.0 + 0.4 * jnp.sin(x) + 0.1 * jnp.cos(y))
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    ms = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(maps={"zp": lambda sigma, H: sigma * H},
                                params={"H": depth_fn})
    grid = Grid((mx, my, ms), mapping=mapping)
    return grid, mx.center * my.center * ms.center


def build_solver_3d(nx, ny, nz, *, coarsen_vertical=True, levels=None,
                    iterations=5):
    """Return a 3-D multigrid-preconditioned mapped solver and its grid."""
    grid, space = build_grid_3d(nx, ny, nz)
    solver = MappedPressureSolver(
        grid, space, iterations=iterations, weights={"sigma": 1.0 / DSQR},
        preconditioner="multigrid", multigrid_levels=levels,
        multigrid_coarsen_vertical=coarsen_vertical)
    return solver, grid, space


def realized_shapes(solver):
    """Return each V-cycle level's (x, y, z) solver-space shape."""
    vcycle = solver._build_vcycle({})
    return [tuple(level.operator.func.__self__._space.shape)
            for level in vcycle.levels]


def mean_free_rhs_3d(grid, space):
    """Return a smooth mean-free rhs on the 3-D pressure space."""
    rhs = grid.create_field(
        space,
        init=lambda x, y, sigma: jnp.exp(
            -((x - 3.0) ** 2 + (y - 3.0) ** 2 + (sigma - 0.5) ** 2) * 3.0))
    return rhs - _computational_mean(rhs)


def test_coarsen_vertical_defaults_true_and_is_stored():
    # the ratified GM-D9 default: the knob is True unless overridden
    solver, *_ = build_solver_3d(16, 16, 16)
    assert solver._multigrid_coarsen_vertical is True
    off, *_ = build_solver_3d(16, 16, 16, coarsen_vertical=False)
    assert off._multigrid_coarsen_vertical is False


def test_default_coarsens_the_vertical_column():
    # GM-D9: the mapped column halves alongside the horizontals,
    # 32^3 -> 16^3 -> 8^3 -> 4^3 (the spike's realized hierarchy)
    solver, *_ = build_solver_3d(32, 32, 32)
    assert realized_shapes(solver) == [
        (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]


def test_semicoarsening_knob_keeps_the_vertical_full():
    # multigrid_coarsen_vertical=False restores the pre-flip
    # semicoarsened shapes: the sigma column stays at 32 at every level
    solver, *_ = build_solver_3d(32, 32, 32, coarsen_vertical=False)
    assert realized_shapes(solver) == [
        (32, 32, 32), (16, 16, 32), (8, 8, 32), (4, 4, 32)]


@pytest.mark.parametrize(
    ("nz", "expected"),
    [pytest.param(24, [(16, 16, 24), (8, 8, 12), (4, 4, 6)], id="nz24"),
     pytest.param(20, [(16, 16, 20), (8, 8, 10), (4, 4, 5)], id="nz20")])
def test_default_indivisible_nz_degrades_and_solves(nz, expected):
    # GD-2 graceful degradation at the SOLVER level: an indivisible n_z
    # stops coarsening the column at the floor/parity wall (24 -> 12 -> 6
    # stops since 6 // 2 = 3 < 4; 20 -> 10 -> 5 stops at the odd 5) while
    # the horizontals keep halving — the solve assembles and runs, no
    # error on the knob
    solver, grid, space = build_solver_3d(16, 16, nz, iterations=12)
    assert realized_shapes(solver) == expected
    rhs = mean_free_rhs_3d(grid, space)
    p = jax.jit(solver.solve)(rhs)
    assert bool(jnp.all(jnp.isfinite(p.data)))


@pytest.mark.parametrize("coarsen_vertical", [True, False])
def test_multigrid_converges_under_both_coarsenings(coarsen_vertical):
    # both knob values drive the mapped Poisson below tolerance in a
    # small PCG budget (the line smoother is kept at every level either
    # way — a point smoother stalls under full coarsening too, spike 2)
    solver, grid, space = build_solver_3d(
        16, 16, 16, coarsen_vertical=coarsen_vertical, iterations=20)
    rhs = mean_free_rhs_3d(grid, space)
    p = jax.jit(solver.solve)(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-7


def test_full_and_semi_coarsening_agree_on_the_solution():
    # the coarsening axis is a preconditioner choice, never physics: the
    # full-coarsened and semicoarsened solves converge to the SAME
    # mean-free pressure (GD-1 parity, single device). Both solvers ride
    # the SAME grid so a single rhs field is valid for either.
    grid, space = build_grid_3d(16, 16, 16)
    kw = {"iterations": 25, "weights": {"sigma": 1.0 / DSQR},
          "preconditioner": "multigrid"}
    full = MappedPressureSolver(grid, space,
                                multigrid_coarsen_vertical=True, **kw)
    semi = MappedPressureSolver(grid, space,
                                multigrid_coarsen_vertical=False, **kw)
    rhs = mean_free_rhs_3d(grid, space)
    p_full = jax.jit(full.solve)(rhs)
    p_semi = jax.jit(semi.solve)(rhs)
    scale = float(jnp.abs(p_semi.data).max())
    assert float(jnp.abs((p_full - p_semi).data).max()) / scale < 1e-6


# ================================================================
#  Perf-guard gap A: the vertical-line smoother on EVERY level
# ================================================================
def test_every_level_smoother_is_a_vertical_line_jacobi():
    # a silent swap to a point (isotropic) smoother looks faster per
    # V-cycle but stalls on steep mapped columns (the budget guard
    # below). Pin the anisotropy smoother on every level built the
    # production way — host-side, no compile.
    solver, _, _ = build_solver(iterations=6, levels=3)
    vcycle = solver._build_vcycle({})
    assert vcycle.levels
    assert all(isinstance(level.smoother, VerticalLineJacobi)
               for level in vcycle.levels)


def test_steep_mapped_multigrid_converges_within_budget():
    # perf-guard gap A convergence budget: the vertical-line V-cycle
    # drives the steep (depth-ratio-9) mapped Poisson below tolerance in
    # a small PCG budget. A point-smoother swap stalls here (rel ~1,
    # measured), so this fails loudly if the anisotropy smoother is lost.
    solver, grid, space = build_solver(
        iterations=13, levels=3, depth_fn=steep_depth)
    rhs = mean_free_rhs(grid, space)
    p = jax.jit(solver.solve)(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-7


# ================================================================
#  Perf-guard gap B: the "auto" knob resolves end to end
# ================================================================
def _resolved_level_methods(solver):
    """Resolve every level smoother's kernel against the backend."""
    vcycle = solver._build_vcycle({})
    assert vcycle.levels
    assert all(level.smoother.method == "auto"
               for level in vcycle.levels)
    return {_resolve_tridiagonal_method(level.smoother.method)
            for level in vcycle.levels}


@pytest.mark.skipif(_ON_GPU, reason="pcr is the off-GPU resolution")
def test_auto_method_wires_to_pcr_off_gpu():
    # end-to-end wiring: the model default "auto" reaches every level's
    # smoother and resolves (host-side) to the portable pcr kernel off a
    # GPU — a silent loss costs 9-18x on the CG solve
    solver, _, _ = build_solver(iterations=3, levels=3)
    assert solver._multigrid_tridiagonal_method == "auto"
    assert _resolved_level_methods(solver) == {"pcr"}


@pytest.mark.skipif(not _ON_GPU, reason="cusparse needs a CUDA GPU")
def test_auto_method_wires_to_cusparse_on_gpu():
    # the GPU leg of the same wiring: "auto" resolves to the batched
    # cuSPARSE kernel on a CUDA backend (runs on the A100 suite)
    solver, _, _ = build_solver(iterations=3, levels=3)
    assert solver._multigrid_tridiagonal_method == "auto"
    assert _resolved_level_methods(solver) == {"cusparse"}
