"""Optional ``tolerance`` convergence break of ``ConjugateGradient``.

Prefix-mirrored shard of ``spatial.operators.krylov`` covering the
opt-in masked-scan stopping criterion (module docstring, *Optional
convergence break*): the construction guards, the early stop on the
measure-weighted true relative residual, the bit-for-bit neutrality of
``tolerance=None`` and of a tolerance too tiny to fire, the zero-RHS
immediate convergence, the traced iteration count under ``jit``, the
preconditioner / ``project_mean`` / ``projection`` branches, and the
reverse- and forward-mode differentiability of the *truncated* algorithm
(the repo invariant: ``jax.grad`` through the solve matches a central
finite difference). Self-contained per the AGENTS oversized-module rule:
the small SPD builders are duplicated from ``test_krylov.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.spatial.operators.base import Identity
from fridom.spatial.operators.krylov import ConjugateGradient
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
    return fr.spatial.Grid((mx, my), device_ids=(0,))


def spectral_pieces(grid, space, sign=-1.0):
    """Return ``(A, exact_inverse)`` for ``sign * Laplacian``."""
    lap = laplacian()
    op = sign * lap
    solve = SpectralSolve(op, grid, space)
    transform = solve.transform
    coeff = transform.codomain(space.bare)
    symbol = op.eigenvalues(grid, coeff)

    def apply(field):
        return transform.backward(symbol(transform.forward(field)))

    return apply, solve


def helmholtz_pieces(grid, space, c=0.2):
    """Return ``(A, exact_inverse)`` for the SPD ``I - c*Laplacian``.

    ``c = 0.2`` keeps the operator well conditioned, so unpreconditioned
    CG contracts monotonically (an early stop lands at a definite step).
    """
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


def stencil_helmholtz(theta):
    """Return the SPD stencil operator ``f - theta * Lap(f)``.

    A matrix-free operator that closes over the scalar ``theta`` (so a
    ``jax.grad`` can differentiate through the operator), ghost-consuming
    and SPD for ``theta > 0``.
    """
    return lambda f: f - theta * (f.diff("x").diff("x")
                                  + f.diff("y").diff("y"))


def weighted_norm(field):
    """Return the measure-weighted norm ``sqrt(<f, f>)`` the solver uses."""
    return float(jnp.sqrt(jnp.sum((field * field).integrate().data)))


# ================================================================
#  Construction guards (house style like ``iterations``)
# ================================================================
def test_rejects_boolean_tolerance():
    with pytest.raises(TypeError, match="tolerance must be an int"):
        ConjugateGradient(lambda f: f, iterations=1, tolerance=True)


def test_rejects_non_numeric_tolerance():
    with pytest.raises(TypeError, match="tolerance must be an int"):
        ConjugateGradient(lambda f: f, iterations=1, tolerance="1e-6")


@pytest.mark.parametrize(
    "value",
    [pytest.param(0.0, id="zero"), pytest.param(-1e-6, id="negative")],
)
def test_rejects_non_positive_tolerance(value):
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        ConjugateGradient(lambda f: f, iterations=1, tolerance=value)


def test_tolerance_property_defaults_to_1e_8():
    plain = ConjugateGradient(lambda f: f, iterations=3)
    assert plain.tolerance == 1e-8
    fixed = ConjugateGradient(lambda f: f, iterations=3, tolerance=None)
    assert fixed.tolerance is None
    tol = ConjugateGradient(lambda f: f, iterations=3, tolerance=1e-6)
    assert tol.tolerance == 1e-6


# ================================================================
#  The early stop: fewer steps, residual below the threshold
# ================================================================
def test_loose_tolerance_stops_before_the_budget():
    # the well-conditioned unpreconditioned Helmholtz contracts
    # monotonically; a loose tolerance stops well before the cap and
    # the returned true residual honours the requested bound
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = helmholtz_pieces(grid, space)
    tol = 1e-2
    cap = 40
    cg = ConjugateGradient(apply_a, iterations=cap, tolerance=tol)
    x, info = cg.solve(rhs)
    assert bool(jnp.isfinite(x.data).all())
    assert 1 < int(info["iterations"]) < cap
    assert float(info["residual_norm"]) <= tol * weighted_norm(rhs)


def test_tighter_tolerance_takes_more_steps():
    # the stopping step is monotone in the requested accuracy
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = helmholtz_pieces(grid, space)
    steps = []
    for tol in (1e-2, 1e-4, 1e-6):
        cg = ConjugateGradient(apply_a, iterations=40, tolerance=tol)
        _, info = cg.solve(rhs)
        steps.append(int(info["iterations"]))
    assert steps[0] < steps[1] < steps[2]


# ================================================================
#  Neutrality: tolerance=None is verbatim; a sub-floor tolerance
#  that never fires reproduces it bit for bit
# ================================================================
def test_tiny_tolerance_never_fires_matches_none_bitwise_unpre():
    # tol far below any achievable residual never fires, so every scan
    # step runs the identical real CG arithmetic; the extra dots and the
    # counter do not feed the iterates, so the solution is bit-for-bit
    # the tolerance=None result (this pins perf/bitwise neutrality)
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = helmholtz_pieces(grid, space)
    # fixed-iteration mode: pinned for determinism
    none_cg = ConjugateGradient(apply_a, iterations=5, tolerance=None)
    tiny_cg = ConjugateGradient(apply_a, iterations=5, tolerance=1e-30)
    assert np.array_equal(np.asarray(none_cg(rhs).data),
                          np.asarray(tiny_cg(rhs).data))


def test_tiny_tolerance_never_fires_matches_none_bitwise_pre():
    # the same neutrality with a preconditioner in the loop
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    # fixed-iteration mode: pinned for determinism
    none_cg = ConjugateGradient(apply_a, preconditioner=exact,
                                iterations=4, tolerance=None)
    tiny_cg = ConjugateGradient(apply_a, preconditioner=exact,
                                iterations=4, tolerance=1e-30)
    assert np.array_equal(np.asarray(none_cg(rhs).data),
                          np.asarray(tiny_cg(rhs).data))


# ================================================================
#  Zero right-hand side: converge immediately, no NaN
# ================================================================
def test_zero_rhs_converges_immediately_without_nan():
    # bb == 0, so the threshold is zero and the residual (already zero
    # after the peel) clears it on the first scan step: the solve holds
    # zero, never dividing tiny by tiny
    grid = build_grid()
    rhs = 0.0 * rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = spectral_pieces(grid, space, sign=-1.0)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=8, tolerance=1e-6,
                           project_mean=True)
    x, info = cg.solve(rhs)
    assert bool(jnp.isfinite(x.data).all())
    assert float(jnp.abs(x.data).max()) == 0.0
    assert float(info["residual_norm"]) == 0.0
    # only the always-run peel executed; the scan did no real step
    assert int(info["iterations"]) == 1


# ================================================================
#  Under jit: the iteration count is a traced 0-d array
# ================================================================
def test_jitted_tolerance_solve_reports_a_traced_iteration_count():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, _ = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, iterations=40, tolerance=1e-3)

    @jax.jit
    def run(d):
        x, info = cg.solve(rhs.with_data(d))
        return x.data, info["iterations"]

    x_d, k = run(rhs.data)
    assert bool(jnp.isfinite(x_d).all())
    # data-dependent break -> a 0-d jax array, not a Python int
    assert isinstance(k, jax.Array)
    assert jnp.ndim(k) == 0
    assert 1 <= int(k) <= 40


# ================================================================
#  The nullspace branches compose with the tolerance
# ================================================================
def test_tolerance_with_project_mean_solves_the_singular_poisson():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = spectral_pieces(grid, space, sign=-1.0)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=10, tolerance=1e-2,
                           project_mean=True)
    x, info = cg.solve(rhs)
    assert int(info["iterations"]) < 10  # the exact inverse fires early
    x_exact = exact(rhs - rhs.mean())
    assert float(jnp.abs(x.data - x_exact.data).max()) < 1e-8
    assert float(jnp.abs(jnp.sum(x.mean().data))) < 1e-12


def test_tolerance_with_custom_projection_matches_project_mean():
    # projection=(f - f.mean()) is the project_mean special case; with a
    # tolerance both fire at the same converged solution
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = spectral_pieces(grid, space, sign=-1.0)
    mean_cg = ConjugateGradient(apply_a, preconditioner=exact,
                                iterations=10, tolerance=1e-8,
                                project_mean=True)
    proj_cg = ConjugateGradient(apply_a, preconditioner=exact,
                                iterations=10, tolerance=1e-8,
                                projection=lambda f: f - f.mean())
    assert np.array_equal(np.asarray(mean_cg(rhs).data),
                          np.asarray(proj_cg(rhs).data))


def test_tolerance_with_exact_preconditioner_fires_on_the_first_step():
    grid = build_grid()
    rhs = rich_rhs(grid)
    space = rhs.function_space
    apply_a, exact = helmholtz_pieces(grid, space)
    cg = ConjugateGradient(apply_a, preconditioner=exact,
                           iterations=12, tolerance=1e-8)
    x, info = cg.solve(rhs)
    # the exact inverse converges in the peel; every scan step is a
    # cond no-op, so the count is the peel alone
    assert int(info["iterations"]) == 1
    assert float(info["residual_norm"]) <= 1e-8 * weighted_norm(rhs)
    assert float(jnp.abs(x.data - exact(rhs).data).max()) < 1e-10


# ================================================================
#  Differentiability (AGENTS.md policy): grad of the truncated
#  algorithm is finite and FD-matched; forward mode agrees
# ================================================================
def test_grad_through_tolerance_solve_wrt_rhs_matches_fd():
    grid = build_grid()
    rhs = rich_rhs(grid)
    cg = ConjugateGradient(stencil_helmholtz(0.02), iterations=30,
                           tolerance=1e-6)

    def loss(scale):
        return jnp.sum(cg(scale * rhs).data ** 2)

    x0 = jnp.asarray(2.0, dtype=jnp.float64)
    grad = float(jax.grad(loss)(x0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-6
    fd = (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)


def test_grad_through_tolerance_solve_wrt_operator_param_matches_fd():
    # differentiate a scale closed over the operator callable: the
    # masked scan differentiates the actual truncated algorithm, so the
    # reverse gradient is exact to FD precision (no custom_vjp, no IFT)
    grid = build_grid()
    rhs = rich_rhs(grid)

    def loss(theta):
        cg = ConjugateGradient(stencil_helmholtz(theta), iterations=30,
                               tolerance=1e-6)
        return jnp.sum(cg(rhs).data ** 2)

    theta0 = jnp.asarray(0.02, dtype=jnp.float64)
    grad = float(jax.grad(loss)(theta0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-6
    fd = (float(loss(theta0 + h)) - float(loss(theta0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)


def test_forward_mode_jvp_through_tolerance_solve_agrees():
    # both AD modes stay alive (no custom_vjp forecloses jvp)
    grid = build_grid()
    rhs = rich_rhs(grid)

    def loss(theta):
        cg = ConjugateGradient(stencil_helmholtz(theta), iterations=30,
                               tolerance=1e-6)
        return jnp.sum(cg(rhs).data ** 2)

    theta0 = jnp.asarray(0.02, dtype=jnp.float64)
    _, jvp = jax.jvp(loss, (theta0,),
                     (jnp.asarray(1.0, dtype=jnp.float64),))
    assert np.isfinite(float(jvp))
    np.testing.assert_allclose(float(jvp), float(jax.grad(loss)(theta0)),
                               rtol=1e-6)
