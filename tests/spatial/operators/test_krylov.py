"""Tests for the ``ConjugateGradient`` preconditioned CG solver core.

The manufactured SPD problems are the spectral-symbol Laplacians the
exact ``SpectralSolve`` diagonalizes: a Helmholtz shift ``I - c*Lap``
(no nullspace) and the pure Poisson ``-Lap`` (constants nullspace,
handled by ``project_mean``). Applying the operator in physical space
is the transform sandwich ``backward @ symbol @ forward`` (the pattern
of ``test_spectral_solve.test_solution_solves_the_equation``), so the
same ``SpectralSolve`` is its exact inverse and hence the ideal
preconditioner. The gates of CS-D2 / plan section 5: PCG matches the
exact solve to rounding in few iterations, the exact preconditioner
converges in one step while the unpreconditioned iteration lags, the
fixed-iteration solve jit-compiles once across right-hand-side values
and is reverse-mode differentiable, the residual norm decreases with
iterations, and the measure-weighted inner product is halo-clean under
a forced-4-device decomposition.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.spatial.operators.base import Identity
from fridom.spatial.operators.krylov import (
    ConjugateGradient,
    _guarded_ratio,
)
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve


def laplacian():
    """Build the 2D spectral Laplacian symbol (``x`` and ``y``)."""
    return (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
            + SpectralDerivative()["y"] @ SpectralDerivative()["y"])


def build_grid(nx=16, ny=16):
    """Build a 2D periodic grid on the unit square."""
    mx = fr.spatial.meshes.IntervalMesh(nx, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(ny, (0.0, 1.0), name="y")
    return fr.spatial.Grid((mx, my))


def spectral_pieces(grid, space, sign=-1.0):
    """Return ``(A, exact_inverse)`` for ``sign * Laplacian``.

    ``A`` applies the operator in physical space (transform sandwich);
    the returned ``SpectralSolve`` is its exact diagonal inverse.
    """
    lap = laplacian()
    op = sign * lap
    solve = SpectralSolve(op, grid, space)
    transform = solve.transform
    coeff = transform.codomain(space.bare)
    symbol = op.eigenvalues(grid, coeff)

    def apply(field):
        return transform.backward(symbol(transform.forward(field)))

    return apply, solve


def helmholtz_pieces(grid, space, c=0.05):
    """Return ``(A, exact_inverse)`` for the SPD ``I - c*Laplacian``."""
    op = Identity() + (-c) * laplacian()
    solve = SpectralSolve(op, grid, space)
    transform = solve.transform
    coeff = transform.codomain(space.bare)
    symbol = op.eigenvalues(grid, coeff)

    def apply(field):
        return transform.backward(symbol(transform.forward(field)))

    return apply, solve


def rich_rhs(grid):
    """Build a multi-mode (peaked) right-hand side."""
    return grid.create_field(
        init=lambda x, y: jnp.exp(
            -((x - 0.5) ** 2 + (y - 0.4) ** 2) * 30.0))


# ================================================================
#  Manufactured SPD problems match the exact spectral solve
# ================================================================
def test_helmholtz_matches_the_exact_solve():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact, iterations=2)
    x = cg(rhs)
    x_exact = exact(rhs)
    assert x.function_space.bare is x_exact.function_space.bare
    assert float(jnp.abs(x.data - x_exact.data).max()) < 1e-10


def test_pure_poisson_with_mean_projection():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = spectral_pieces(grid, space, sign=-1.0)
    cg = ConjugateGradient(
        apply_a, preconditioner=exact, iterations=3, project_mean=True)
    x = cg(rhs)
    # SpectralSolve's k=0 gauge is the mean-free solution; compare
    # against the same mean-free right-hand side it solves.
    x_exact = exact(rhs - rhs.mean())
    assert float(jnp.abs(x.data - x_exact.data).max()) < 1e-10
    # the solution carries no mean (the pinned nullspace gauge)
    assert float(jnp.abs(jnp.sum(x.mean().data))) < 1e-12


# ================================================================
#  Preconditioner effect
# ================================================================
def test_exact_preconditioner_converges_in_one_step():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact, iterations=1)
    _, info = cg.solve(rhs)
    assert float(info["residual_norm"]) < 1e-10


def test_preconditioning_beats_no_preconditioner():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    pre = ConjugateGradient(apply_a, preconditioner=exact, iterations=2)
    raw = ConjugateGradient(apply_a, iterations=2)
    _, pre_info = pre.solve(rhs)
    _, raw_info = raw.solve(rhs)
    # the exact preconditioner reaches rounding; the unpreconditioned
    # iteration is still far from converged after the same budget
    assert float(pre_info["residual_norm"]) < float(
        raw_info["residual_norm"])
    assert float(raw_info["residual_norm"]) > 1e-6


def test_unpreconditioned_defaults_to_identity():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, iterations=1)
    assert cg.preconditioner is None
    # a single unpreconditioned step still produces a finite field
    x = cg(rhs)
    assert bool(jnp.all(jnp.isfinite(x.data)))


# ================================================================
#  Convergence sanity: the residual decreases with iterations
# ================================================================
def test_residual_decreases_with_iterations():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = spectral_pieces(grid, space, sign=-1.0)
    norms = []
    for n in (1, 3, 6):
        cg = ConjugateGradient(apply_a, iterations=n, project_mean=True)
        _, info = cg.solve(rhs)
        norms.append(float(info["residual_norm"]))
    assert norms[0] > norms[1] > norms[2]


# ================================================================
#  Fixed-iteration purity: one trace across values, differentiable
# ================================================================
def test_solve_compiles_once_across_rhs_values(compile_counter):
    grid = build_grid()
    rhs1 = rich_rhs(grid)
    rhs2 = rhs1 * 3.0
    space = rhs1.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact, iterations=3)

    @jax.jit
    def run(rhs):
        return cg(rhs).data

    warm = run(rhs1)
    compile_counter.reset()
    a = run(rhs1)
    assert compile_counter.count == 0  # cache hit
    b = run(rhs2)
    assert compile_counter.count == 0  # values swept, one trace
    assert jnp.allclose(a, warm)
    assert jnp.allclose(b, 3.0 * warm)


def test_grad_through_the_solve_is_finite():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact, iterations=2)

    def loss(scale):
        return jnp.sum(cg(scale * rhs).data ** 2)

    grad = jax.grad(loss)(2.0)
    assert bool(jnp.isfinite(grad))
    assert float(grad) != 0.0


# ================================================================
#  The scanned recurrence (ROADMAP 3.6): O(1) trace and compile
# ================================================================
def unrolled_reference(cg, rhs):
    """Run the pre-3.6 fully-unrolled recurrence, for comparison."""
    b = cg._project(rhs)
    x = 0.0 * b
    r = b
    z = cg._project(cg._precondition(r))
    p = z
    rz = cg._dot(r, z)
    for _ in range(cg.iterations):
        x, r, p, rz = cg._step(x, r, p, rz)
    return cg._project(x)


@pytest.mark.parametrize("iterations", [1, 2, 5, 12, 30])
def test_scanned_solve_matches_the_unrolled_recurrence(iterations):
    # the scan is a trace-structure change only. It is not bitwise:
    # XLA fuses and FMA-contracts a scan body differently from
    # straight-line code, which moves the last bits (measured ~1 ulp,
    # 5.6e-17 absolute at 12/30 iterations, far below the 6e-15
    # mapped-flat identity gate). The teardown/rebuild of the carry
    # through .data / with_data is itself exactly bitwise.
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=iterations)
    scanned = np.asarray(cg(rhs).data)
    reference = np.asarray(unrolled_reference(cg, rhs).data)
    assert np.allclose(scanned, reference, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("iterations", [3, 12, 60])
def test_operator_is_traced_twice_regardless_of_iterations(iterations):
    # the O(1)-trace gate in its most direct form: the loop body is
    # traced ONCE (plus the peeled first iteration), so the opaque
    # operator is applied a constant number of times at trace time no
    # matter how many iterations the recurrence runs.
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    calls = []

    def counting(field):
        calls.append(field)
        return apply_a(field)

    cg = ConjugateGradient(counting, preconditioner=exact,
                           iterations=iterations)
    cg(rhs)
    assert len(calls) == 2  # the peel, then the single scan body


def test_exchanges_per_iteration_are_unchanged_by_the_scan(monkeypatch):
    # The exchange gate. A stencil operator (unlike the spectral one)
    # really consumes ghosts, so the recurrence pays halo exchanges.
    #
    # The canonical halo state the scan body rebuilds is exactly the
    # state field arithmetic already produced -- every CG iterate is
    # the output of a +/- and therefore claims zero valid ghosts even
    # in the unrolled form -- so the rebuild re-declares what the
    # iterates already had and cannot add a sync. The gate: the scan
    # body pays the SAME exchanges per iteration as the unrolled loop.
    original = fr.spatial.Grid.sync
    calls = []

    def counting(self, field, boundary_data=None):
        calls.append(field)
        return original(self, field, boundary_data)

    def apply_a(f):
        # an SPD stencil Helmholtz: I - c * Lap, ghost-consuming
        return f - 0.02 * (f.diff("x").diff("x")
                           + f.diff("y").diff("y"))

    def fresh(iterations):
        # a fresh grid/rhs per measurement: the synced-ghost memo is
        # identity-keyed, so a shared rhs would carry a memoized
        # exchange across runs and skew the counts
        grid = build_grid()
        return (rich_rhs(grid),
                ConjugateGradient(apply_a, iterations=iterations))

    def unrolled_syncs(iterations):
        rhs, cg = fresh(iterations)
        calls.clear()
        monkeypatch.setattr(fr.spatial.Grid, "sync", counting)
        unrolled_reference(cg, rhs)
        monkeypatch.undo()
        return len(calls)

    # the unrolled reference is a straight line in the iteration count
    u4, u8 = unrolled_syncs(4), unrolled_syncs(8)
    per_iteration = (u8 - u4) // 4
    assert per_iteration > 0  # the operator really does exchange
    assert u8 == u4 + 4 * per_iteration

    # the scanned form: record the exchanges of each traced _step
    original_step = ConjugateGradient._step
    steps = []

    def recording_step(self, x, r, p, rz):
        before = len(calls)
        out = original_step(self, x, r, p, rz)
        steps.append(len(calls) - before)
        return out

    traced = {}
    for n in (4, 8):
        rhs, cg = fresh(n)
        calls.clear()
        steps.clear()
        monkeypatch.setattr(fr.spatial.Grid, "sync", counting)
        monkeypatch.setattr(ConjugateGradient, "_step", recording_step)
        cg(rhs)
        monkeypatch.undo()
        # _step is traced exactly twice: the peel, then the one body
        assert len(steps) == 2
        # and the body costs exactly what an unrolled iteration cost
        assert steps[1] == per_iteration
        traced[n] = len(calls)

    # the trace itself no longer grows with the iteration budget...
    assert traced[4] == traced[8]
    # ...while the runtime total is unchanged: the trace holds
    # setup + peel + ONE body, and the body re-executes (n-1) times
    assert traced[4] + per_iteration * (4 - 2) == u4
    assert traced[8] + per_iteration * (8 - 2) == u8



def test_hlo_size_is_constant_in_the_iteration_count():
    # the compile-cost gate: an unrolled loop grew the HLO linearly
    # (145 675 lines at 300 iterations, 28 s of compile); the scanned
    # body is emitted once, so the program size no longer depends on
    # the iteration budget.
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)

    def hlo_lines(iterations):
        cg = ConjugateGradient(apply_a, preconditioner=exact,
                               iterations=iterations)
        lowered = jax.jit(lambda d: cg(rhs.with_data(d)).data).lower(
            rhs.data)
        return lowered.as_text().count("\n")

    lines = {k: hlo_lines(k) for k in (12, 60, 300)}
    # the scanned body is emitted once: 5x the budget, zero growth
    assert lines[300] == lines[60]
    # small budgets may lower a few dozen lines differently — jax
    # flips its constants-as-arguments choice on an estimated module
    # size (inline dense<...> blobs vs %arg lifting), which is a
    # representation change, not per-iteration growth; bound it
    assert abs(lines[12] - lines[60]) < 100


def test_grad_flows_through_a_long_scanned_solve():
    # the reason the loop is a lax.scan and not a lax.fori_loop:
    # reverse-mode must keep flowing through the whole recurrence,
    # including a long one that lives entirely inside the scan body
    # (fori_loop is not reverse-mode differentiable; scan is).
    #
    # Unpreconditioned on purpose. Over-iterating an *exact*
    # preconditioner drives the residual to ~1e-17 rather than to the
    # exact zero _guarded_ratio tests for, so the ratios divide
    # tiny-by-tiny: finite forward, NaN in reverse. That predates the
    # scan (the unrolled recurrence NaNs at the same iteration counts)
    # and is a property of the fixed-iteration design, not of the
    # loop form.
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = spectral_pieces(grid, space, sign=-1.0)
    cg = ConjugateGradient(apply_a, iterations=25, project_mean=True)

    def loss(scale):
        return jnp.sum(cg(scale * rhs).data ** 2)

    grad = jax.grad(loss)(2.0)
    assert bool(jnp.isfinite(grad))
    assert float(grad) != 0.0


# ================================================================
#  Construction guards
# ================================================================
def test_rejects_non_callable_operator():
    with pytest.raises(TypeError, match="field-to-field callable"):
        ConjugateGradient(object(), iterations=1)


def test_rejects_non_callable_preconditioner():
    with pytest.raises(TypeError, match="preconditioner"):
        ConjugateGradient(
            lambda f: f, preconditioner=object(), iterations=1)


def test_rejects_non_integer_iterations():
    with pytest.raises(TypeError, match="iterations must be an int"):
        ConjugateGradient(lambda f: f, iterations=2.0)


def test_rejects_boolean_iterations():
    with pytest.raises(TypeError, match="iterations must be an int"):
        ConjugateGradient(lambda f: f, iterations=True)


def test_rejects_non_positive_iterations():
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        ConjugateGradient(lambda f: f, iterations=0)


def test_properties_expose_the_configuration():
    apply_a = lambda f: f  # noqa: E731
    pre = lambda f: f  # noqa: E731
    cg = ConjugateGradient(
        apply_a, preconditioner=pre, iterations=4, project_mean=True)
    assert cg.operator is apply_a
    assert cg.preconditioner is pre
    assert cg.iterations == 4
    assert cg.project_mean is True


# ================================================================
#  Initial guess
# ================================================================
def test_explicit_initial_guess_matches_zero_start():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact, iterations=3)
    x_zero = cg(rhs)
    # starting from the exact solution: the residual is already zero,
    # so the solve holds it there and reproduces the same field
    x_guess = cg(rhs, x0=exact(rhs))
    assert float(jnp.abs(x_zero.data - x_guess.data).max()) < 1e-10


# ================================================================
#  Exact convergence under fixed iterations (the guarded ratios)
# ================================================================
def test_zero_rhs_stays_finite_and_returns_zero():
    # rz == 0 from iteration one: the guarded ratios make every
    # iteration an exact no-op instead of a 0/0 NaN
    grid = build_grid()
    rhs = 0.0 * rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = spectral_pieces(grid, space, sign=-1.0)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=5, project_mean=True)
    x, info = cg.solve(rhs)
    assert bool(jnp.isfinite(x.data).all())
    assert float(jnp.abs(x.data).max()) == 0.0
    assert float(info["residual_norm"]) == 0.0


def test_over_iterating_an_exact_preconditioner_stays_finite():
    # the exact inverse converges in one step; the remaining
    # iterations divide (near-)zero by (near-)zero and must remain
    # exact no-ops on the converged solution (module docstring)
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=12)
    x, _info = cg.solve(rhs)
    assert bool(jnp.isfinite(x.data).all())
    want = exact(rhs)
    scale = float(jnp.abs(want.data).max())
    assert float(jnp.abs(x.data - want.data).max()) < 1e-12 * scale


def test_guarded_ratio_is_division_off_the_zero_branch():
    num = jnp.asarray(3.0)
    den = jnp.asarray(4.0)
    assert float(_guarded_ratio(num, den)) == float(num / den)
    assert float(_guarded_ratio(num, jnp.asarray(0.0))) == 0.0
    grad = jax.grad(
        lambda d: _guarded_ratio(jnp.asarray(1.0), d) ** 2)
    assert bool(jnp.isfinite(grad(jnp.asarray(0.0))))


# ================================================================
#  Halo-cleanliness: the weighted dot is decomposition-invariant
# ================================================================
@pytest.mark.multi_device
def test_solution_is_device_count_invariant():
    # If a halo or stagger-pad slot leaked into the CG inner products,
    # the sharded (multi-device) solve would diverge from the
    # single-device one. Routing every dot through ``integrate`` (true
    # DOFs, ``grid.measure``) keeps them halo-clean. The comparison is
    # to tight rounding, not bitwise: the cross-shard reductions sum in
    # a different order, so the last bits legitimately differ.
    def run(device_ids):
        mx = fr.spatial.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
        my = fr.spatial.meshes.IntervalMesh(32, (0.0, 1.0), name="y")
        grid = fr.spatial.Grid((mx, my), device_ids=device_ids)
        rhs = rich_rhs(grid)
        space = rhs.function_space
        apply_a, _ = spectral_pieces(grid, space, sign=-1.0)
        cg = ConjugateGradient(
            apply_a, iterations=5, project_mean=True)
        return np.asarray(cg(rhs).data)

    many = run(None)
    one = run((0,))
    assert np.allclose(many, one, rtol=0.0, atol=1e-11)
