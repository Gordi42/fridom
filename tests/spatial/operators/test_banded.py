"""Tests for the lifted banded (tridiagonal) solve primitive.

Covers the Neumann second-difference band assembly against a dense
numpy reference (including its constant-mode nullspace), the batched
apply along a chosen storage axis, the batched solve as the exact
inverse of the apply, and the per-column Thomas kernel
(``tridiagonal_solve_along_axis``) against a dense per-column solve
(batched, axis-agnostic, reverse-mode differentiable).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.operators.banded import (
    apply_along_axis,
    second_difference_matrix,
    solve_along_axis,
    tridiagonal_solve_along_axis,
)

N = 12


def _dense_tridiagonal(lower, diag, upper):
    """Dense (N, N) matrix from a single column's three bands."""
    return (np.diag(np.asarray(diag))
            + np.diag(np.asarray(upper)[:-1], 1)
            + np.diag(np.asarray(lower)[1:], -1))


def _random_bands(rng, shape):
    """Diagonally dominant symmetric-ish bands (non-singular columns)."""
    diag = jnp.asarray(rng.uniform(4.0, 6.0, shape))
    lower = jnp.asarray(rng.uniform(-1.0, 1.0, shape))
    upper = jnp.asarray(rng.uniform(-1.0, 1.0, shape))
    # Neumann ends: lower[0] and upper[-1] unused
    lower = lower.at[0].set(0.0)
    upper = upper.at[-1].set(0.0)
    return lower, diag, upper


def dense_reference(nodes):
    """Neumann second-difference / dz^2 as a plain numpy matrix."""
    nodes = np.asarray(nodes).reshape(-1)
    size = nodes.shape[0]
    dz = nodes[1] - nodes[0]
    main = np.full(size, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    return (np.diag(main) + np.diag(np.ones(size - 1), 1)
            + np.diag(np.ones(size - 1), -1)) / dz ** 2


# ================================================================
#  Band assembly
# ================================================================
def test_second_difference_matches_dense_reference():
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    band = second_difference_matrix(jnp.asarray(nodes))
    assert band.shape == (N, N)
    assert np.allclose(np.asarray(band), dense_reference(nodes))


def test_second_difference_annihilates_the_constant_mode():
    # the zero-flux (Neumann) boundary rows preserve constants
    nodes = jnp.asarray((np.arange(N) + 0.5) * (2.0 / N))
    band = second_difference_matrix(nodes)
    const = jnp.ones((N,))
    assert float(jnp.abs(band @ const).max()) < 1e-10


def test_second_difference_flattens_a_column_node_field():
    # the assembly reads the 1D line, so a column-shaped (N, 1) node
    # field flattens to the same band
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    column = jnp.asarray(nodes).reshape(N, 1)
    band = second_difference_matrix(column)
    assert band.shape == (N, N)
    assert np.allclose(np.asarray(band), dense_reference(nodes))


# ================================================================
#  Batched apply / solve
# ================================================================
@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_apply_along_axis_matches_a_manual_contraction(axis_index):
    rng = np.random.default_rng(0)
    operator = jnp.asarray(rng.standard_normal((N, N)))
    data = jnp.asarray(rng.standard_normal((N, N, N)))
    out = apply_along_axis(operator, data, axis_index)
    expected = np.moveaxis(
        np.tensordot(np.asarray(operator), np.asarray(data),
                     axes=([1], [axis_index])),
        0, axis_index)
    assert out.shape == data.shape
    assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize("axis_index", [0, 1])
def test_solve_is_the_inverse_of_apply(axis_index):
    rng = np.random.default_rng(1)
    # a well-conditioned SPD-ish system
    a = rng.standard_normal((N, N))
    system = jnp.asarray(a @ a.T + N * np.eye(N))
    x = jnp.asarray(rng.standard_normal((N, N)))
    rhs = apply_along_axis(system, x, axis_index)
    recovered = solve_along_axis(system, rhs, axis_index)
    assert np.allclose(np.asarray(recovered), np.asarray(x))


def test_solve_matches_numpy_linalg_solve():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((N, N))
    system = jnp.asarray(a @ a.T + N * np.eye(N))
    rhs = jnp.asarray(rng.standard_normal((N, 4)))
    solved = solve_along_axis(system, rhs, 0)
    expected = np.linalg.solve(np.asarray(system), np.asarray(rhs))
    assert np.allclose(np.asarray(solved), expected)


# ================================================================
#  Per-column Thomas solve (distinct tridiagonal per column)
# ================================================================
def test_tridiagonal_solve_matches_dense_per_column():
    # a distinct tridiagonal per off-axis column: the Thomas kernel
    # must reproduce a dense jnp.linalg.solve column by column
    rng = np.random.default_rng(0)
    shape = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solved = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    for i in range(shape[1]):
        for j in range(shape[2]):
            dense = _dense_tridiagonal(
                lower[:, i, j], diag[:, i, j], upper[:, i, j])
            expected = np.linalg.solve(
                dense, np.asarray(rhs[:, i, j]))
            assert np.allclose(np.asarray(solved[:, i, j]), expected)


def test_tridiagonal_solve_is_the_inverse_of_the_band_apply():
    # T x = b then solve(T, b) recovers x, batched over columns
    rng = np.random.default_rng(1)
    shape = (N, 5)
    lower, diag, upper = _random_bands(rng, shape)
    x = jnp.asarray(rng.standard_normal(shape))
    # apply the tridiagonal per column (Neumann ends respected)
    up_shift = jnp.concatenate(
        [x[1:], jnp.zeros((1, shape[1]))], axis=0)
    lo_shift = jnp.concatenate(
        [jnp.zeros((1, shape[1])), x[:-1]], axis=0)
    b = diag * x + upper * up_shift + lower * lo_shift
    recovered = tridiagonal_solve_along_axis(lower, diag, upper, b, 0)
    assert np.allclose(np.asarray(recovered), np.asarray(x))


@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_tridiagonal_solve_is_axis_agnostic(axis_index):
    # solving along any storage axis matches solving along axis 0
    rng = np.random.default_rng(3)
    base = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, base)
    rhs = jnp.asarray(rng.standard_normal(base))
    ref = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    lo = jnp.moveaxis(lower, 0, axis_index)
    di = jnp.moveaxis(diag, 0, axis_index)
    up = jnp.moveaxis(upper, 0, axis_index)
    r = jnp.moveaxis(rhs, 0, axis_index)
    out = tridiagonal_solve_along_axis(lo, di, up, r, axis_index)
    assert np.allclose(
        np.asarray(jnp.moveaxis(out, axis_index, 0)), np.asarray(ref))


def test_tridiagonal_solve_broadcasts_the_bands():
    # a column-uniform band (shape (N, 1)) broadcasts over the batch
    rng = np.random.default_rng(4)
    lower, diag, upper = _random_bands(rng, (N, 1))
    rhs = jnp.asarray(rng.standard_normal((N, 6)))
    solved = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    dense = _dense_tridiagonal(lower[:, 0], diag[:, 0], upper[:, 0])
    expected = np.linalg.solve(dense, np.asarray(rhs))
    assert np.allclose(np.asarray(solved), expected)


def test_tridiagonal_solve_is_reverse_mode_differentiable():
    # scan Thomas (no custom_vjp): jax.grad flows and matches an FD
    rng = np.random.default_rng(5)
    shape = (N, 3)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))

    def loss(scale):
        x = tridiagonal_solve_along_axis(
            lower, diag, upper, scale * rhs, 0)
        return jnp.sum(x ** 2)

    grad = jax.grad(loss)(2.0)
    eps = 1e-6
    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(grad))
    assert abs(float(grad) - float(fd)) <= 1e-4 * abs(float(fd))
