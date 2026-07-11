"""Sine/Cosine (DST/DCT) tests with synthesis-matrix oracles."""
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.dealias import degree
from fridom.spatial.operators.trig import Cosine, Sine
from fridom.spatial.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def bounded():
    mesh = IntervalMesh(N, (0.0, 1.0), periodic=False, name="x")
    return Grid((mesh,)), mesh


def _dirichlet_center(mesh):
    return mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)


def _dirichlet_inner(mesh):
    return mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)


def _neumann_center(mesh):
    return mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)


def _neumann_outer(mesh):
    return mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN)


def _sine_matrix(x, modes):
    """Synthesis oracle: column k-1 is sin(k pi x / L), L = 1."""
    k = np.arange(1, modes + 1)
    return np.sin(np.pi * np.outer(np.asarray(x), k))


def _cosine_matrix(x, modes):
    """Synthesis oracle: column k is cos(k pi x / L), L = 1."""
    k = np.arange(modes)
    return np.cos(np.pi * np.outer(np.asarray(x), k))


# ================================================================
#  DST-II (Dirichlet Center origin, n modes k = 1..n)
# ================================================================
@pytest.mark.parametrize("k0", [1, 3, N],
                         ids=["first", "mid", "top"])
def test_dst2_single_modes_give_delta_coefficients(bounded, k0):
    grid, mesh = bounded
    space = _dirichlet_center(mesh)
    x = grid.evaluation_nodes(space).data
    f = grid.create_field(space, data=jnp.sin(k0 * jnp.pi * x))
    coeff = Sine(grid).forward(f)
    assert coeff.function_space.bare is mesh.sine(space)
    assert coeff.shape == (N,)
    expected = jnp.zeros(N).at[k0 - 1].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_dst2_backward_matches_the_synthesis_matrix(bounded):
    grid, mesh = bounded
    space = _dirichlet_center(mesh)
    x = grid.evaluation_nodes(space).data
    rng = np.random.default_rng(0)
    b = rng.standard_normal(N)
    coeff = grid.create_field(mesh.sine(space), data=jnp.asarray(b))
    back = Sine(grid).backward(coeff)
    assert back.function_space.bare is space
    assert jnp.allclose(back.data, _sine_matrix(x, N) @ b,
                        atol=1e-13)


def test_dst2_round_trip(bounded):
    grid, mesh = bounded
    space = _dirichlet_center(mesh)
    f = grid.random.normal(space, seed=11)
    op = Sine(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-13)


def test_dst2_complex_data(bounded):
    grid, mesh = bounded
    space = _dirichlet_center(mesh).as_complex()
    x = grid.evaluation_nodes(space).data
    data = (1 + 2j) * jnp.sin(2 * jnp.pi * x)
    f = grid.create_field(space, data=data)
    op = Sine(grid)
    coeff = op.forward(f)
    assert jnp.issubdtype(coeff.dtype, jnp.complexfloating)
    expected = jnp.zeros(N, dtype=complex).at[1].set(1 + 2j)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)
    assert jnp.allclose(op.backward(coeff).data, data, atol=1e-14)


# ================================================================
#  DST-I (Dirichlet Inner origin, n - 1 modes k = 1..n-1)
# ================================================================
def test_dst1_shape_and_delta(bounded):
    grid, mesh = bounded
    space = _dirichlet_inner(mesh)
    x = grid.evaluation_nodes(space).data
    f = grid.create_field(space, data=jnp.sin(2 * jnp.pi * x))
    coeff = Sine(grid).forward(f)
    assert coeff.function_space.bare is mesh.sine(space)
    assert coeff.shape == (N - 1,)
    expected = jnp.zeros(N - 1).at[1].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_dst1_backward_matches_the_synthesis_matrix(bounded):
    grid, mesh = bounded
    space = _dirichlet_inner(mesh)
    x = grid.evaluation_nodes(space).data
    rng = np.random.default_rng(1)
    b = rng.standard_normal(N - 1)
    coeff = grid.create_field(mesh.sine(space),
                              data=jnp.asarray(b))
    back = Sine(grid).backward(coeff)
    assert jnp.allclose(back.data, _sine_matrix(x, N - 1) @ b,
                        atol=1e-13)


def test_dst1_round_trip(bounded):
    grid, mesh = bounded
    space = _dirichlet_inner(mesh)
    f = grid.random.normal(space, seed=12)
    op = Sine(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-13)


# ================================================================
#  DCT-II (Neumann Center origin, n modes k = 0..n-1)
# ================================================================
def test_dct2_constant_and_mode_delta(bounded):
    grid, mesh = bounded
    space = _neumann_center(mesh)
    x = grid.evaluation_nodes(space).data
    f = grid.create_field(
        space, data=0.5 + jnp.cos(3 * jnp.pi * x))
    coeff = Cosine(grid).forward(f)
    assert coeff.function_space.bare is mesh.cosine(space)
    assert coeff.shape == (N,)
    expected = jnp.zeros(N).at[0].set(0.5).at[3].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_dct2_backward_matches_the_synthesis_matrix(bounded):
    grid, mesh = bounded
    space = _neumann_center(mesh)
    x = grid.evaluation_nodes(space).data
    rng = np.random.default_rng(2)
    a = rng.standard_normal(N)
    coeff = grid.create_field(mesh.cosine(space),
                              data=jnp.asarray(a))
    back = Cosine(grid).backward(coeff)
    assert back.function_space.bare is space
    assert jnp.allclose(back.data, _cosine_matrix(x, N) @ a,
                        atol=1e-13)


def test_dct2_round_trip(bounded):
    grid, mesh = bounded
    space = _neumann_center(mesh)
    f = grid.random.normal(space, seed=13)
    op = Cosine(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-13)


# ================================================================
#  DCT-I (Neumann Outer origin, n + 1 modes k = 0..n)
# ================================================================
@pytest.mark.parametrize("k0", [0, 3, N],
                         ids=["constant", "mid", "top"])
def test_dct1_single_modes_give_delta_coefficients(bounded, k0):
    grid, mesh = bounded
    space = _neumann_outer(mesh)
    x = jnp.linspace(0.0, 1.0, N + 1)  # the n + 1 face nodes
    f = grid.create_field(space, data=jnp.cos(k0 * jnp.pi * x))
    coeff = Cosine(grid).forward(f)
    assert coeff.function_space.bare is mesh.cosine(space)
    assert coeff.shape == (N + 1,)
    expected = jnp.zeros(N + 1).at[k0].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_dct1_backward_matches_the_synthesis_matrix(bounded):
    grid, mesh = bounded
    space = _neumann_outer(mesh)
    x = jnp.linspace(0.0, 1.0, N + 1)
    rng = np.random.default_rng(3)
    a = rng.standard_normal(N + 1)
    coeff = grid.create_field(mesh.cosine(space),
                              data=jnp.asarray(a))
    back = Cosine(grid).backward(coeff)
    assert back.function_space.bare is space
    assert jnp.allclose(back.data, _cosine_matrix(x, N + 1) @ a,
                        atol=1e-13)


def test_dct1_round_trip(bounded):
    grid, mesh = bounded
    space = _neumann_outer(mesh)
    f = grid.random.normal(space, seed=14)
    op = Cosine(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-13)


def test_dct1_complex_data(bounded):
    grid, mesh = bounded
    space = _neumann_outer(mesh).as_complex()
    x = jnp.linspace(0.0, 1.0, N + 1)
    data = (1 + 2j) * jnp.cos(2 * jnp.pi * x)
    f = grid.create_field(space, data=data)
    op = Cosine(grid)
    coeff = op.forward(f)
    assert jnp.issubdtype(coeff.dtype, jnp.complexfloating)
    expected = jnp.zeros(N + 1, dtype=complex).at[2].set(1 + 2j)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)
    assert jnp.allclose(op.backward(coeff).data, data, atol=1e-14)


# ================================================================
#  Origin scope (iteration 1)
# ================================================================
def test_sine_rejects_outer_dirichlet_origins(bounded):
    grid, mesh = bounded
    space = mesh.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    f = grid.create_field(space)
    with pytest.raises(SpaceMismatchError, match="DST"):
        Sine(grid).forward(f)


def test_cosine_rejects_inner_neumann_origins(bounded):
    grid, mesh = bounded
    space = mesh.nodal(NodeSet.INNER, bc=BC.NEUMANN)
    with pytest.raises(SpaceMismatchError, match="DCT"):
        Cosine(grid).codomain(space)


def test_sine_rejects_bc_free_origins(bounded):
    grid, mesh = bounded
    f = grid.create_field(mesh.center)
    with pytest.raises(SpaceMismatchError):
        Sine(grid).forward(f)


def test_sine_rejects_periodic_meshes():
    mesh = IntervalMesh(N, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    f = grid.create_field(mesh.center)
    with pytest.raises(SpaceMismatchError):
        Sine(grid).forward(f)


# ================================================================
#  Padded variants
# ================================================================
def test_padded_sine_backward_and_exact_trim(bounded):
    grid, mesh = bounded
    space = _dirichlet_center(mesh)
    x = grid.evaluation_nodes(space).data
    f = grid.create_field(space, data=jnp.sin(2 * jnp.pi * x))
    plain = Sine(grid)
    padded = Sine(grid, pad=degree(2))
    coeff = plain.forward(f)

    fine = padded.backward(coeff)
    fine_mesh = mesh.refined(Fraction(3, 2))
    fine_space = fine_mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert fine.function_space.bare is fine_space
    assert fine.shape == (12,)
    x_fine = grid.evaluation_nodes(fine_space).data
    assert jnp.allclose(fine.data, jnp.sin(2 * jnp.pi * x_fine),
                        atol=1e-14)

    trimmed = padded.forward(fine)
    assert trimmed.function_space is coeff.function_space
    assert jnp.allclose(trimmed.data, coeff.data, atol=1e-14)


def test_padded_top_mode_survives_the_pad_trim_pair(bounded):
    grid, mesh = bounded
    space = _dirichlet_center(mesh)
    padded = Sine(grid, pad=degree(2))
    top = jnp.zeros(N).at[N - 1].set(1.0)
    coeff = grid.create_field(mesh.sine(space), data=top)
    round_trip = padded.forward(padded.backward(coeff))
    assert jnp.allclose(round_trip.data, top, atol=1e-14)


def test_cosine_rejects_periodic_meshes():
    mesh = IntervalMesh(N, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    with pytest.raises(SpaceMismatchError, match="no DCT"):
        Cosine(grid).codomain(mesh.center)


def test_padded_cosine_pad_trim_round_trip(bounded):
    grid, mesh = bounded
    space = _neumann_center(mesh)
    padded = Cosine(grid, pad=degree(2))
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.standard_normal(N))
    coeff = grid.create_field(mesh.cosine(space), data=a)
    fine = padded.backward(coeff)
    assert fine.shape == (12,)
    round_trip = padded.forward(fine)
    assert round_trip.function_space is coeff.function_space
    assert jnp.allclose(round_trip.data, a, atol=1e-13)


def test_padded_dct1_backward_and_exact_trim(bounded):
    grid, mesh = bounded
    space = _neumann_outer(mesh)
    x = jnp.linspace(0.0, 1.0, N + 1)
    f = grid.create_field(space, data=jnp.cos(2 * jnp.pi * x))
    plain = Cosine(grid)
    padded = Cosine(grid, pad=degree(2))
    coeff = plain.forward(f)

    fine = padded.backward(coeff)
    fine_mesh = mesh.refined(Fraction(3, 2))
    fine_space = fine_mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    assert fine.function_space.bare is fine_space
    assert fine.shape == (13,)
    x_fine = jnp.linspace(0.0, 1.0, 13)
    assert jnp.allclose(fine.data, jnp.cos(2 * jnp.pi * x_fine),
                        atol=1e-14)

    trimmed = padded.forward(fine)
    assert trimmed.function_space is coeff.function_space
    assert jnp.allclose(trimmed.data, coeff.data, atol=1e-14)
