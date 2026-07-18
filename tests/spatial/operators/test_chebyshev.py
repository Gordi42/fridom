"""Chebyshev-transform tests (synthesis oracle: cos(k arccos))."""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.chebyshev import Chebyshev
from fridom.spatial.operators.dealias import degree
from fridom.spatial.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def cheb():
    mesh = ChebyshevMesh(N, (-1.0, 1.0), name="z")
    return Grid((mesh,), device_ids=(0,)), mesh


def _xi():
    """Ascending-x Lobatto nodes on [-1, 1] (storage order)."""
    return -np.cos(np.pi * np.arange(N + 1) / N)


def _chebyshev_matrix(xi, modes):
    """Synthesis oracle: column k is T_k(xi)."""
    k = np.arange(modes)
    return np.cos(np.outer(np.arccos(np.asarray(xi)), k))


# ================================================================
#  Forward/backward against the synthesis oracle
# ================================================================
@pytest.mark.parametrize("k0", [0, 1, 3, N],
                         ids=["T0", "T1", "T3", "Tn"])
def test_chebyshev_polynomials_give_delta_coefficients(cheb, k0):
    grid, mesh = cheb
    xi = _xi()
    values = np.cos(k0 * np.arccos(xi))
    f = grid.create_field(mesh.lobatto, data=jnp.asarray(values))
    coeff = Chebyshev(grid).forward(f)
    assert coeff.function_space.bare is mesh.chebyshev(mesh.lobatto)
    assert coeff.shape == (N + 1,)
    expected = jnp.zeros(N + 1).at[k0].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_backward_matches_the_synthesis_matrix(cheb):
    grid, mesh = cheb
    rng = np.random.default_rng(3)
    a = rng.standard_normal(N + 1)
    coeff = grid.create_field(mesh.chebyshev(mesh.lobatto),
                              data=jnp.asarray(a))
    back = Chebyshev(grid).backward(coeff)
    assert back.function_space.bare is mesh.lobatto
    assert jnp.allclose(back.data, _chebyshev_matrix(_xi(), N + 1) @ a,
                        atol=1e-13)


def test_round_trip(cheb):
    grid, mesh = cheb
    f = grid.random.normal(mesh.lobatto, seed=21)
    op = Chebyshev(grid)
    back = op.backward(op.forward(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-13)


def test_mapped_extent_keeps_the_reference_coefficients():
    # the transform sees index space: coefficients are those of the
    # affine pullback onto [-1, 1] regardless of the physical extent
    mesh = ChebyshevMesh(N, (0.0, 2.0), name="z")
    grid = Grid((mesh,), device_ids=(0,))
    xi = _xi()
    f = grid.create_field(mesh.lobatto,
                          data=jnp.asarray(2 * xi ** 2 - 1))
    coeff = Chebyshev(grid).forward(f)
    expected = jnp.zeros(N + 1).at[2].set(1.0)
    assert jnp.allclose(coeff.data, expected, atol=1e-14)


def test_polynomial_exactness(cheb):
    grid, mesh = cheb
    xi = _xi()
    f = grid.create_field(
        mesh.lobatto,
        data=jnp.asarray(xi ** 3 - 2 * xi ** 2 + 0.5))
    op = Chebyshev(grid)
    back = op.backward(op.forward(f))
    assert jnp.allclose(back.data, f.data, atol=1e-14)


# ================================================================
#  Origin scope
# ================================================================
def test_bc_structured_origins_are_designed_for(cheb):
    grid, mesh = cheb
    space = mesh.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    f = grid.create_field(space)
    with pytest.raises(SpaceMismatchError, match="Shen"):
        Chebyshev(grid).forward(f)


def test_interval_center_origins_raise():
    mesh = IntervalMesh(N, (0.0, 1.0), periodic=False, name="x")
    grid = Grid((mesh,), device_ids=(0,))
    f = grid.create_field(mesh.center)
    with pytest.raises(SpaceMismatchError, match="Gauss-Lobatto"):
        Chebyshev(grid).forward(f)


# ================================================================
#  Padded variant: refinement is designed-for on ChebyshevMesh
# ================================================================
def test_padded_chebyshev_raises_at_construction(cheb):
    grid, _ = cheb
    with pytest.raises(NotImplementedError, match="refined"):
        Chebyshev(grid, pad=degree(2))


def test_interval_outer_origins_raise():
    # right node set but wrong mesh family: the factory error path
    mesh = IntervalMesh(N, (0.0, 1.0), periodic=False, name="x")
    grid = Grid((mesh,), device_ids=(0,))
    f = grid.create_field(mesh.outer)
    with pytest.raises(SpaceMismatchError, match="no Chebyshev"):
        Chebyshev(grid).forward(f)
