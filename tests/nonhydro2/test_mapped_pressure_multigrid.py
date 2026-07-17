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

TWO_PI = 2.0 * np.pi
DSQR = 0.25
NZ = 8


# ================================================================
#  Self-contained builders (each shard duplicates its helpers)
# ================================================================
def depth(x):
    """Smooth periodic water depth H(x) (steep-ish, ratio 3)."""
    return 1.0 + 0.5 * jnp.sin(x)


def build_grid(nx=16, nz=NZ, *, device_ids=None):
    """Terrain-following 2D grid ``zp = sigma * H(x)`` (x periodic)."""
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(maps={"zp": lambda sigma, H: sigma * H},
                                params={"H": depth})
    return Grid((mx, ms), mapping=mapping, device_ids=device_ids), mx, ms


def build_solver(nx=16, *, iterations, levels, device_ids=None):
    """Return a multigrid-preconditioned mapped solver and its grid."""
    grid, mx, ms = build_grid(nx, device_ids=device_ids)
    space = mx.center * ms.center
    solver = MappedPressureSolver(
        grid, space, iterations=iterations,
        weights={"sigma": 1.0 / DSQR},
        preconditioner="multigrid", multigrid_levels=levels)
    return solver, grid, space


def mean_free_rhs(grid, space):
    """Return a smooth, decomposition-invariant mean-free rhs."""
    rhs = grid.create_field(
        space,
        init=lambda x, sigma: jnp.exp(
            -((x - 3.0) ** 2 + (sigma - 0.5) ** 2) * 3.0))
    return rhs - rhs.mean()


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
    # recursion (16 -> 8 -> 4 on x, sigma kept full)
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
