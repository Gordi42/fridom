"""Tests for the diagonal ``SpectralSolve`` elliptic solver.

The typed port of the inline ``test_spectral_poisson`` pattern: a 2D
Poisson solve reproducing the analytic solution to machine precision,
the mean-free (``k = 0``) nullspace gauge and its ``where_zero``
override, a Helmholtz shift (no nullspace), and the residual check
(applying the Laplacian to the solution recovers the mean-free rhs).
"""
import jax.numpy as jnp
import pytest

import fridom.framework2 as fr
from fridom.framework2.grid.operators.spectral import SpectralDerivative
from fridom.framework2.grid.operators.spectral_solve import SpectralSolve
from fridom.framework2.grid.operators.symbol import Symbol


def laplacian_2d():
    return (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
            + SpectralDerivative()["y"] @ SpectralDerivative()["y"])


@pytest.fixture
def grid_2d():
    mx = fr.grid.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
    my = fr.grid.meshes.IntervalMesh(32, (0.0, 2.0), name="y")
    return fr.grid.Grid((mx, my))


# ================================================================
#  The Poisson solve reproduces the analytic solution
# ================================================================
def test_reproduces_the_poisson_solution(grid_2d):
    grid = grid_2d
    u_exact = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    lam = (4 * jnp.pi) ** 2 + jnp.pi ** 2
    rhs = grid.create_field(
        init=lambda x, y: -lam * jnp.sin(4 * jnp.pi * x)
        * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    u = solve(rhs)
    assert u.function_space.bare is u_exact.function_space.bare
    # measured ~1.4e-15, the same as the inline test_spectral_poisson
    assert float(jnp.abs(u.data - u_exact.data).max()) < 1e-10
    # the k = 0 gauge is the mean-free solution
    assert float(jnp.abs(u.mean().data.ravel()[0])) < 1e-13


def test_solve_alias_matches_call(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    assert jnp.array_equal(solve.solve(rhs).data, solve(rhs).data)


# ================================================================
#  The solution actually solves the equation (residual check)
# ================================================================
def test_solution_solves_the_equation():
    mx = fr.grid.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    laplacian = SpectralDerivative()["x"] @ SpectralDerivative()["x"]
    rhs = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x)
        + 0.25 * jnp.cos(6 * jnp.pi * x))
    solve = SpectralSolve(laplacian, grid, rhs.function_space)
    u = solve(rhs)
    # applying the same elliptic symbol to the solution recovers the
    # (mean-free) right-hand side
    u_hat = solve.transform.forward(u)
    sym = laplacian.eigenvalues(grid, u_hat.function_space.bare)
    residual = solve.transform.backward(sym(u_hat))
    assert float(jnp.abs(residual.data - rhs.data).max()) < 1e-10


# ================================================================
#  The nullspace gauge and its ``where_zero`` override
# ================================================================
def test_where_zero_sets_the_nullspace_gauge():
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    laplacian = SpectralDerivative()["x"] @ SpectralDerivative()["x"]
    rhs = grid.create_field(init=lambda x: jnp.cos(2 * jnp.pi * x))
    default = SpectralSolve(laplacian, grid, rhs.function_space)
    shifted = SpectralSolve(laplacian, grid, rhs.function_space,
                            where_zero=1.0)
    # the k = 0 diagonal is regularized to 0 (default) vs 1
    assert default.inverse_symbol.data.ravel()[0] == 0.0
    assert shifted.inverse_symbol.data.ravel()[0] == 1.0


def test_helmholtz_shift_has_no_nullspace():
    # (d_xx - lambda) with lambda != 0 inverts everywhere: no k = 0 zero
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    dxx = SpectralDerivative()["x"] @ SpectralDerivative()["x"]
    space = grid.create_field(init=lambda x: x).function_space
    t = grid.dispatch.resolve("transform", space.bare)
    coeff = t.codomain(space.bare)
    helmholtz = dxx.eigenvalues(grid, coeff) - Symbol(
        coeff, 3.0 * jnp.ones(coeff.shape))
    inv = helmholtz.inverse()
    # the k = 0 entry is -1/3, not a regularized zero
    assert float(inv.data.ravel()[0].real) == pytest.approx(-1.0 / 3.0)


# ================================================================
#  Properties
# ================================================================
def test_properties_expose_the_transform_and_inverse(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    coeff = solve.transform.codomain(rhs.function_space.bare)
    assert solve.inverse_symbol.space is coeff
