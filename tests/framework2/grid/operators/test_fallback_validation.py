"""Validation gates for the graded ``Fallback`` operator (stage F3).

The gates F1's ``test_fallback.py`` did NOT cover (plan
``notes/framework2/fallback_operator_plan.md`` section 5):

1. jit single-compile — sweeping field VALUES retraces exactly once.
2. Multi-device forced-4 — the graded operator on a bounded axis kept
   undistributed (``layout="local"``) is bitwise-identical to the
   single-device run.
3. Second-order coverage — WENO-3 graded (K=1): NaN-poison
   interior-only gate + interior convergence rate ~= 3.
4. Graceful boundary behavior — finite, close-not-exact wall values,
   error bounded and monotonically reducing (no NaN/Inf/oscillation).

Everything is driven through the shipped ``graded_reconstruction`` /
``Fallback`` API (no F2 knob needed).
"""
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.fallback import graded_reconstruction


# ================================================================
#  Helpers (mirror F1's test_fallback.py construction)
# ================================================================
def sin_cell_averages(a, b, n):
    """Exact cell averages of sin(2 pi y) over n cells of [a, b]."""
    edges = np.linspace(a, b, n + 1)
    anti = -np.cos(2.0 * np.pi * edges) / (2.0 * np.pi)
    dx = (b - a) / n
    return (anti[1:] - anti[:-1]) / dx


def bounded_field(n, order, data=None):
    """Build a CellAvg field on a bounded mesh with WENO-`order` halo.

    Pinned to a single device (``device_ids=(0,)``): these are numerical
    order / jit gates, so the bounded axis must stay undistributed
    regardless of the ambient device count (the forced-4 suite would
    otherwise shard the lone axis -- a path the graded operator does not
    yet support; the sharded-but-local scenario is the decomposition-
    suite gate in ``decomposition/test_fallback_multi_device.py``).
    """
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"y": order // 2 + 1}))
    if data is None:
        data = sin_cell_averages(0.0, 1.0, n)
    f = grid.create_field(mesh.cell_avg, data=jnp.asarray(data))
    return mesh, grid, f


def masked_interior_convergence(order, bias, sizes=(32, 64, 128)):
    """Return max reconstruction error at interior faces per size.

    Sampled away from the critical points of sin (WENO-JS degrades
    there), one value per size.
    """
    op = graded_reconstruction(order, bias)
    errors = []
    for n in sizes:
        mesh, grid, f = bounded_field(n, order)
        g = op["y"](f)
        assert g.function_space.bare is mesh.inner
        y = np.asarray(grid.evaluation_nodes(mesh.inner).data)
        err = np.abs(np.asarray(g.data) - np.sin(2.0 * np.pi * y))
        interior = np.zeros_like(y, dtype=bool)
        interior[op.reduced_rows: y.size - op.reduced_rows] = True
        mask = interior & (np.abs(np.cos(2.0 * np.pi * y)) > 0.3)
        errors.append(err[mask].max())
    return op, errors


# ================================================================
#  Gate 1: jit single-compile (static K, static slots)
# ================================================================
def test_graded_traces_once_across_value_sweep(compile_counter):
    # Applying the graded operator inside a jax.jit and sweeping the
    # field VALUES (identical shapes) must retrace exactly once: K and
    # the wall slots are compile-time constants, the interior/boundary
    # split is a static index partition (never a data-dependent where).
    n = 48
    op = graded_reconstruction(5, "left")
    mesh, grid, _ = bounded_field(n, 5)
    rng = np.random.default_rng(21)
    # fields differing only in data values (same space, same shape)
    fields = [
        grid.create_field(mesh.cell_avg,
                          data=jnp.asarray(rng.normal(size=n)))
        for _ in range(4)
    ]

    @jax.jit
    def run(f):
        return op["y"](f)

    # warm the cache on the first field (traces once), then sweep the
    # remaining VALUES and assert zero retraces -- the static K / static
    # wall slots make the graph value-independent.
    run(fields[0]).data.block_until_ready()
    compile_counter.reset()
    results = [run(f) for f in fields[1:]]
    jax.block_until_ready([r.data for r in results])
    assert compile_counter.count == 0
    # sanity: the sweep genuinely produced distinct outputs
    assert not np.array_equal(np.asarray(results[0].data),
                              np.asarray(results[1].data))


def test_graded_weno3_traces_once(compile_counter):
    # same guarantee at the K=1 rung count
    n = 40
    op = graded_reconstruction(3, "right")
    mesh, grid, _ = bounded_field(n, 3)
    rng = np.random.default_rng(4)
    fields = [
        grid.create_field(mesh.cell_avg,
                          data=jnp.asarray(rng.normal(size=n)))
        for _ in range(3)
    ]

    @jax.jit
    def run(f):
        return op["y"](f)

    run(fields[0]).data.block_until_ready()
    compile_counter.reset()
    results = [run(f) for f in fields[1:]]
    jax.block_until_ready([r.data for r in results])
    assert compile_counter.count == 0


# Gate 2 (multi-device forced-4 bitwise identity) lives in
# tests/framework2/grid/decomposition/test_fallback_multi_device.py --
# the forced-4 CI job (.github/workflows/tests.yml:35-42) only globs the
# decomposition suite, so a forced-4 gate must live there to be run
# genuinely sharded.


# ================================================================
#  Gate 3: second-order coverage (WENO-3 graded, K=1)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
def test_weno3_nan_poison_gate_proves_interior_only(bias):
    # K=1 twin of F1's WENO-5 poison gate: poison every exterior halo
    # cell with NaN, claim the ghosts valid so the consumption-side
    # sync leaves them, and assert the true output stays finite -- any
    # exterior read would propagate a NaN.
    n = 16
    mesh, grid, f = bounded_field(n, 3)
    width = grid.decomposition.halo["y"]
    assert width == 2  # WENO-3 halo
    storage = f._data
    poisoned = storage.at[:width].set(jnp.nan)
    poisoned = poisoned.at[storage.shape[0] - width:].set(jnp.nan)
    f._data = poisoned
    f._halo_valid = HaloSpec({"y": width})

    op = graded_reconstruction(3, bias)["y"]
    result = op(f)
    assert result.function_space.bare is mesh.inner
    assert bool(jnp.all(jnp.isfinite(result.data)))


@pytest.mark.parametrize("bias", ["left", "right"])
def test_weno3_interior_convergence_is_third_order(bias):
    op, errors = masked_interior_convergence(3, bias)
    assert op.reduced_rows == 1  # K = halo(2) - slack(1)
    slopes = [np.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert min(slopes) > 2.6  # design order 3 at interior faces


# ================================================================
#  Gate 4: graceful boundary behavior (no blow-up, no oscillation)
# ================================================================
@pytest.mark.parametrize("order", [5, 3])
def test_wall_rows_degrade_gracefully(order):
    # On a smooth field the wall-adjacent output values are finite and
    # close (not exact) to the analytic face value; the error stays
    # bounded and REDUCES under refinement (order reduction degrades
    # gracefully -- no NaN/Inf, no oscillatory blow-up).
    sizes = (64, 128, 256)
    op = graded_reconstruction(order, "left")
    K = op.reduced_rows
    wall_errors = []
    for n in sizes:
        mesh, grid, f = bounded_field(n, order)
        g = op["y"](f)
        assert bool(jnp.all(jnp.isfinite(g.data)))  # no NaN/Inf
        y = np.asarray(grid.evaluation_nodes(mesh.inner).data)
        err = np.abs(np.asarray(g.data) - np.sin(2.0 * np.pi * y))
        wall = np.concatenate([err[:K], err[-K:]])
        # close (not exact): the reduced wall value tracks the analytic
        # face value to a sane absolute bound on an O(1) field
        assert wall.max() < 0.1
        # bounded relative to the interior: finite ratio, no blow-up
        assert np.all(np.isfinite(err))
        wall_errors.append(wall.max())
    # graceful: the wall error monotonically reduces (>= ~1st order,
    # the wall-adjacent upwind rung), it does not grow or oscillate
    for coarse, fine in pairwise(wall_errors):
        assert fine < coarse
    slopes = [np.log2(wall_errors[k] / wall_errors[k + 1])
              for k in range(len(sizes) - 1)]
    assert min(slopes) > 0.8  # graceful reduction, no blow-up
