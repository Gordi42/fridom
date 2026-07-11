"""Numerical oracles for the VerticalDiffusion tridiagonal kernel.

Covers ``apply`` and ``solve`` against a dense numpy reference
(Neumann zero-flux second-difference / dz^2 scaled by kappa), the
constant-mode preservation of the zero-flux boundary rows, a 2D field
diffused along a chosen axis, the exactly-merged kappa-summed operator
(two closures collapse to ONE solve with the summed coefficient),
``dt_gamma`` as a traced positional (a gamma sweep shares one
compile), and the field-valued-kappa rejection.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.implicit import VerticalDiffusion
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 16


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mz():
    return IntervalMesh(N, (0.0, 1.0), name="z")


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 2.0), name="x")


@pytest.fixture
def grid(mz):
    return Grid((mz,))


def field(grid, space, data, name="b"):
    return grid.create_field(space, data=jnp.asarray(data), name=name)


def dense_operator(grid, space, axis, kappa):
    """Build the (M,M) Neumann second-difference along ``axis``."""
    nodes = np.asarray(
        grid.evaluation_nodes(space, axis).data).reshape(-1)
    size = nodes.shape[0]
    dz = nodes[1] - nodes[0]
    main = np.full(size, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    d2 = (np.diag(main) + np.diag(np.ones(size - 1), 1)
          + np.diag(np.ones(size - 1), -1)) / dz ** 2
    return kappa * d2


def kappa_const(value):
    def kappa(_module, _state, _ctx, _name):
        return value
    return kappa


# ================================================================
#  apply — the forward L @ state
# ================================================================
def test_apply_matches_dense_reference(grid, mz):
    op = VerticalDiffusion("z", ("b",), kappa_const(0.05))
    b0 = np.cos((np.arange(N) + 0.5) * 3 * np.pi / N)
    state = VectorField({"b": field(grid, mz.center, b0)})
    result = op.apply(None, state, None)["b"]
    reference = dense_operator(grid, mz.center, "z", 0.05) @ b0
    assert np.allclose(np.asarray(result.data), reference, atol=1e-12)


def test_apply_annihilates_the_constant_mode(grid, mz):
    # zero-flux (Neumann) boundary rows: L @ const == 0
    op = VerticalDiffusion("z", ("b",), kappa_const(0.3))
    state = VectorField({"b": field(grid, mz.center, np.full(N, 4.2))})
    result = op.apply(None, state, None)["b"]
    assert np.allclose(np.asarray(result.data), 0.0, atol=1e-10)


def test_apply_rejects_unknown_axis(grid, mz):
    op = VerticalDiffusion("y", ("b",), kappa_const(0.1))
    state = VectorField({"b": field(grid, mz.center, np.zeros(N))})
    with pytest.raises(ValueError, match="not a coordinate"):
        op.apply(None, state, None)


# ================================================================
#  solve — (1 - dt_gamma L) x = rhs
# ================================================================
def test_solve_matches_dense_reference(grid, mz):
    op = VerticalDiffusion("z", ("b",), kappa_const(0.05))
    rhs0 = np.linspace(-1.0, 1.0, N)
    rhs = {"b": field(grid, mz.center, rhs0)}
    dt_gamma = 0.11
    solved = op.solve(None, rhs, dt_gamma, None)["b"]
    matrix = dense_operator(grid, mz.center, "z", 0.05)
    reference = np.linalg.solve(np.eye(N) - dt_gamma * matrix, rhs0)
    assert np.allclose(np.asarray(solved.data), reference, atol=1e-12)


def test_solve_preserves_the_constant_mode(grid, mz):
    # (1 - dt_gamma L) const == const  (L annihilates the constant)
    op = VerticalDiffusion("z", ("b",), kappa_const(0.2))
    rhs = {"b": field(grid, mz.center, np.full(N, 2.5))}
    solved = op.solve(None, rhs, 0.7, None)["b"]
    assert np.allclose(np.asarray(solved.data), 2.5, atol=1e-9)


def test_solve_inverts_apply_at_matching_gamma(grid, mz):
    # x = solve((1 - g L), (1 - g L) x)  round-trips
    op = VerticalDiffusion("z", ("b",), kappa_const(0.05))
    x0 = np.cos((np.arange(N) + 0.5) * 2 * np.pi / N)
    state = VectorField({"b": field(grid, mz.center, x0)})
    applied = op.apply(None, state, None)["b"]
    dt_gamma = 0.1
    rhs_data = x0 - dt_gamma * np.asarray(applied.data)
    rhs = {"b": field(grid, mz.center, rhs_data)}
    solved = op.solve(None, rhs, dt_gamma, None)["b"]
    assert np.allclose(np.asarray(solved.data), x0, atol=1e-10)


# ================================================================
#  A 2D field, diffused along a chosen axis
# ================================================================
def test_apply_along_a_chosen_axis_of_a_2d_field(mx, mz):
    grid = Grid((mx, mz))
    space = mx.center * mz.center
    op = VerticalDiffusion("z", ("b",), kappa_const(0.05))
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, N))
    state = VectorField({"b": field(grid, space, data)})
    result = op.apply(None, state, None)["b"]
    matrix = dense_operator(grid, space, "z", 0.05)
    reference = data @ matrix.T  # apply along axis 1
    assert np.allclose(np.asarray(result.data), reference, atol=1e-12)


# ================================================================
#  The exactly-merged kappa-summed operator (ONE solve)
# ================================================================
def test_merged_operator_solves_with_the_summed_kappa(grid, mz):
    op1 = VerticalDiffusion("z", ("b",), kappa_const(2.0))
    op2 = VerticalDiffusion("z", ("b",), kappa_const(3.5))
    merged = op1.merged_with(op2)          # ONE operator, kappa = 5.5
    reference = VerticalDiffusion("z", ("b",), kappa_const(5.5))
    rhs0 = np.linspace(0.0, 1.0, N)
    rhs = {"b": field(grid, mz.center, rhs0)}
    merged_solved = merged.solve(None, rhs, 0.1, None)["b"]
    summed_solved = reference.solve(None, rhs, 0.1, None)["b"]
    assert np.allclose(np.asarray(merged_solved.data),
                       np.asarray(summed_solved.data), atol=1e-13)


# ================================================================
#  dt_gamma is a traced positional (a gamma sweep shares one compile)
# ================================================================
def test_dt_gamma_sweep_shares_one_compile(grid, mz, compile_counter):
    op = VerticalDiffusion("z", ("b",), kappa_const(0.05))
    rhs0 = np.linspace(-1.0, 1.0, N)
    base = field(grid, mz.center, rhs0)

    @jax.jit
    def solve(component, dt_gamma):
        return op.solve(None, {"b": component}, dt_gamma,
                        None)["b"].data

    solve(base, jnp.asarray(0.1))  # warm the cache
    compile_counter.reset()
    first = solve(base, jnp.asarray(0.1))
    assert compile_counter.count == 0  # already compiled
    second = solve(base, jnp.asarray(0.9))
    assert compile_counter.count == 0  # gamma is dynamic, no retrace
    assert not np.allclose(np.asarray(first), np.asarray(second))


# ================================================================
#  Field-valued kappa is the variable-coefficient follow-up
# ================================================================
def test_field_valued_kappa_is_rejected(grid, mz):
    def kappa(_module, _state, _ctx, _name):
        return field(grid, mz.center, np.ones(N))  # a ScalarField

    op = VerticalDiffusion("z", ("b",), kappa)
    state = VectorField({"b": field(grid, mz.center, np.zeros(N))})
    with pytest.raises(NotImplementedError, match="constant"):
        op.apply(None, state, None)
