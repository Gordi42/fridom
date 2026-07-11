"""Tests for the lifted banded (tridiagonal) solve primitive.

Covers the Neumann second-difference band assembly against a dense
numpy reference (including its constant-mode nullspace), the batched
apply along a chosen storage axis, and the batched solve as the exact
inverse of the apply.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.operators.banded import (
    apply_along_axis,
    second_difference_matrix,
    solve_along_axis,
)

N = 12


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
