"""Tests for the diagonal ``SpectralSolve`` elliptic solver.

The typed port of the inline ``test_spectral_poisson`` pattern: a 2D
Poisson solve reproducing the analytic solution to machine precision,
the mean-free (``k = 0``) nullspace gauge and its ``where_zero``
override, a Helmholtz shift (no nullspace), and the residual check
(applying the Laplacian to the solution recovers the mean-free rhs).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.framework.utils import dtype_real
from fridom.spatial.operators.realized import RealizedComposite
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve, _CastMap
from fridom.spatial.operators.symbol import Symbol


def laplacian_2d():
    return (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
            + SpectralDerivative()["y"] @ SpectralDerivative()["y"])


@pytest.fixture
def grid_2d():
    mx = fr.spatial.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(32, (0.0, 2.0), name="y")
    return fr.spatial.Grid((mx, my))


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


def test_call_is_bitwise_equal_to_imperative_solve(grid_2d):
    # S2 reframe guard: the composite ``backward @ inverse @ forward``
    # must be bitwise-identical to the pre-refactor imperative body
    # ``transform.backward(inverse(transform.forward(rhs)))``
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    t = solve.transform
    imperative = t.backward(solve.inverse_symbol(t.forward(rhs)))
    maxdiff = float(jnp.abs(solve(rhs).data - imperative.data).max())
    assert maxdiff == 0.0
    # and the object *is* the realized-map composition
    assert isinstance(solve.composite, RealizedComposite)


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
    mx = fr.spatial.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
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
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
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
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
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
def test_accepts_a_preassembled_symbol(grid_2d):
    # the metric seam: a solve built from a coefficient-space Symbol
    # (assembled off the operators, e.g. with a traced weight) rather
    # than an operator — must match the operator-built solve
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    from_op = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    coeff = from_op.transform.codomain(rhs.function_space.bare)
    symbol = laplacian_2d().eigenvalues(grid, coeff)
    from_sym = SpectralSolve(symbol, grid, rhs.function_space)
    assert jnp.array_equal(from_sym.inverse_symbol.data,
                           from_op.inverse_symbol.data)
    assert jnp.allclose(from_sym(rhs).data, from_op(rhs).data)


def test_properties_expose_the_transform_and_inverse(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    coeff = solve.transform.codomain(rhs.function_space.bare)
    assert solve.inverse_symbol.space is coeff


# ================================================================
#  Single-precision solve option (Change A)
# ================================================================
def test_single_precision_default_off(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    assert solve.single_precision is False
    # bitwise identical to the imperative full-precision solve
    assert isinstance(solve.composite, RealizedComposite)


def test_single_precision_result_matches_full(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    full = SpectralSolve(laplacian_2d(), grid, rhs.function_space)
    low = SpectralSolve(laplacian_2d(), grid, rhs.function_space,
                        single_precision=True)
    assert low.single_precision is True
    p_full = full(rhs)
    p_low = low(rhs)
    # the solution stays float64 (only the transform pair is reduced)
    assert p_low.dtype == dtype_real()
    rel = float(jnp.linalg.norm(p_low.data - p_full.data)
                / jnp.linalg.norm(p_full.data))
    assert rel < 1e-5


def test_single_precision_runs_the_ffts_in_reduced_precision(grid_2d):
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    low = SpectralSolve(laplacian_2d(), grid, rhs.function_space,
                        single_precision=True)
    hlo = jax.jit(lambda f: low(f).data).lower(rhs).compile().as_text()
    # the rfft lands on complex64 and the irfft on float32 (vs the
    # c128 / f64 of the full-precision solve) — the divide runs single
    fft_lines = [ln for ln in hlo.splitlines() if "fft(" in ln]
    assert fft_lines  # the periodic solve is an rfftn/irfftn pair
    assert any("c64[" in ln for ln in fft_lines)
    assert any("f32[" in ln for ln in fft_lines)
    assert not any("c128[" in ln for ln in fft_lines)


def test_cast_map_casts_field_data(grid_2d):
    grid = grid_2d
    f = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    coeff = SpectralSolve(
        laplacian_2d(), grid, f.function_space).transform.codomain(
        f.function_space.bare)
    cast = _CastMap(coeff, jnp.complex64)
    assert cast.domain is coeff
    assert cast.codomain is coeff
    assert cast.dtype == jnp.dtype(jnp.complex64)
    # conj of a real-linear cast is the cast itself
    assert cast.conj() is cast
    with pytest.raises(NotImplementedError, match="not inverted"):
        cast.inverse()


def test_single_precision_solve_pytree_round_trip(grid_2d):
    # the reduced composite flattens/unflattens (jit/scan friendly)
    grid = grid_2d
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    low = SpectralSolve(laplacian_2d(), grid, rhs.function_space,
                        single_precision=True)
    leaves, treedef = jax.tree_util.tree_flatten(low.composite)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.allclose(rebuilt(rhs).data, low(rhs).data)
