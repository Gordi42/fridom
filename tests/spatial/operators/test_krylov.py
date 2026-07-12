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
