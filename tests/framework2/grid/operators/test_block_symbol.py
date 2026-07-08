"""Tests for ``BlockSymbol`` and ``BlockMatrix.eigenvalues``."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import EigenbasisError, Zero
from fridom.framework2.grid.operators.block_symbol import BlockSymbol
from fridom.framework2.grid.operators.composed import (
    BlockMatrix,
    Divergence,
    Gradient,
    Laplacian,
)
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.spectral import SpectralDerivative

N = 8


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(N, (0.0, 2.0), name="y")


@pytest.fixture
def grid1(mx):
    return Grid((mx,))


@pytest.fixture
def grid2(mx, my):
    return Grid((mx, my))


# ================================================================
#  Assembly, properties and pytree registration
# ================================================================
def test_from_blocks_scatters_and_zeros(mx, my, grid2):
    space = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, space)
    dy = SpectralDerivative()["y"].eigenvalues(grid2, space)
    block = BlockSymbol.from_blocks(
        ((dx, None), (None, dy)), (space, space), (space, space))
    n_modes = block.data.shape[:-2]
    assert block.data.shape == (*n_modes, 2, 2)
    assert jnp.allclose(block.data[..., 0, 0],
                        jnp.broadcast_to(dx.data, n_modes))
    assert jnp.allclose(block.data[..., 1, 1],
                        jnp.broadcast_to(dy.data, n_modes))
    # the ``None`` slots are structural zeros
    assert jnp.allclose(block.data[..., 0, 1], 0.0)
    assert jnp.allclose(block.data[..., 1, 0], 0.0)


def test_from_blocks_needs_a_non_zero_entry(mx):
    space = mx.fourier(origin=mx.center)
    with pytest.raises(ValueError, match="at least one non-zero"):
        BlockSymbol.from_blocks(((None,),), (space,), (space,))


def test_properties_store_bare_space_tuples(mx, my, grid2):
    space = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, space)
    block = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    assert block.in_spaces == (space.bare,)
    assert block.out_spaces == (space.bare,)
    assert block.matrix is block.data


def test_block_symbol_is_a_registered_pytree(mx, grid1):
    space = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid1, space)
    block = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    leaves, treedef = jax.tree_util.tree_flatten(block)
    assert len(leaves) == 1  # only ``_data`` is dynamic
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.in_spaces == block.in_spaces
    assert rebuilt.out_spaces == block.out_spaces
    assert jnp.array_equal(rebuilt.data, block.data)


# ================================================================
#  BlockMatrix.eigenvalues — the 1x1 Laplacian reduction
# ================================================================
def test_1x1_laplacian_block_reduces_to_the_scalar_symbol(grid2, mx, my):
    phys = mx.center * my.center
    block = Laplacian().expand(phys, grid2.dispatch)
    assert len(block.rows) == 1
    assert len(block.rows[0]) == 1
    sym = block.eigenvalues(grid2, phys)
    assert isinstance(sym, BlockSymbol)
    assert sym.data.shape[-2:] == (1, 1)
    # the (0, 0) entry is exactly the scalar Laplacian symbol -k_hat**2
    scalar = block.rows[0][0].eigenvalues(grid2, phys)
    assert jnp.allclose(sym.data[..., 0, 0],
                        jnp.broadcast_to(scalar.data,
                                         sym.data.shape[:-2]))
    # and it is real and negative-definite off the annihilated modes
    assert jnp.allclose(sym.data.imag, 0.0)


def test_gradient_block_symbol_retags_staggered_components(grid2, mx, my):
    phys = mx.center * my.center
    grad = Gradient().expand(phys, grid2.dispatch)
    sym = grad.eigenvalues(grid2, phys)
    # a 2x1 column block: each row retags to its own staggered space
    assert sym.data.shape[-2:] == (2, 1)
    assert sym.out_spaces == (
        (mx.right * my.center).bare, (mx.center * my.right).bare)
    assert sym.in_spaces == (phys.bare,)
    # each entry carries its own per-axis FD phase symbol
    for i in range(2):
        entry = grad.rows[i][0].eigenvalues(grid2, phys)
        assert jnp.allclose(sym.data[..., i, 0],
                            jnp.broadcast_to(entry.data,
                                             sym.data.shape[:-2]))


def test_block_eigenvalues_rejects_the_wrong_arity(grid2, mx, my):
    phys = mx.center * my.center
    grad = Gradient().expand(phys, grid2.dispatch)
    div = Divergence().expand(grad.codomain(phys), grid2.dispatch)
    # div has two columns; querying with one operand space raises
    with pytest.raises(SpaceMismatchError, match="columns queried"):
        div.eigenvalues(grid2, phys)


def test_block_eigenvalues_propagates_eigenbasis_error(grid2, mx, my):
    # a 4th-order FD entry declines its symbol -> propagates
    block = BlockMatrix(((FiniteDifference(4)["x"],),))
    with pytest.raises(EigenbasisError, match="order 2 only"):
        block.eigenvalues(grid2, mx.center * my.center)


# ================================================================
#  Block matmul == the symbol of the composed operator
# ================================================================
def test_div_grad_matmul_equals_the_laplacian_block(grid2, mx, my):
    phys = mx.center * my.center
    grad = Gradient().expand(phys, grid2.dispatch)
    mid = grad.codomain(phys)
    div = Divergence().expand(mid, grid2.dispatch)
    grad_sym = grad.eigenvalues(grid2, phys)
    div_sym = div.eigenvalues(grid2, *mid)
    # the staggered mid tuple threads through: div.in == grad.out
    assert div_sym.in_spaces == grad_sym.out_spaces
    composed = (div @ grad).eigenvalues(grid2, phys)
    assert jnp.allclose((div_sym @ grad_sym).data, composed.data)


def test_2x2_block_matmul_matches_the_composed_operator(grid1, mx):
    # antisymmetric rotation-like block [[0, dx], [-dx, 0]] on 1-D
    fx = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"]
    block = BlockMatrix(((Zero(), dx), ((-1.0) * dx, Zero())),
                        output_names=("a", "b"))
    sym = block.eigenvalues(grid1, fx, fx)
    assert sym.data.shape[-2:] == (2, 2)
    composed = (block @ block).eigenvalues(grid1, fx, fx)
    assert jnp.allclose((sym @ sym).data, composed.data)


def test_matmul_is_the_per_mode_matrix_product(mx, my, grid2):
    sa = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    sb = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, sa)
    dy = SpectralDerivative()["y"].eigenvalues(grid2, sa)
    left = BlockSymbol.from_blocks(
        ((dx, dy), (dy, dx)), (sa, sa), (sb, sb))
    right = BlockSymbol.from_blocks(
        ((dy, dx), (dx, dy)), (sa, sa), (sa, sa))
    product = left @ right
    # in/out tuples thread B.in -> result.in, A.out -> result.out
    assert product.in_spaces == right.in_spaces
    assert product.out_spaces == left.out_spaces
    # the per-mode matrix product, checked against batched matmul
    assert jnp.allclose(product.data,
                        jnp.matmul(left.data, right.data))


def test_matmul_rejects_a_mid_space_mismatch(mx, my, grid2):
    sa = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    sb = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, sa)
    a = BlockSymbol.from_blocks(((dx,),), (sa,), (sb,))
    b = BlockSymbol.from_blocks(((dx,),), (sa,), (sa,))
    # a.in == (sa,), b.out == (sa,) matches; swap to force a mismatch
    with pytest.raises(SpaceMismatchError, match="cannot compose"):
        _ = b @ a  # b.in == (sa,) != a.out == (sb,)


def test_matmul_returns_notimplemented_for_a_foreign_operand(mx, grid1):
    space = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid1, space)
    block = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    assert block.__matmul__(3) is NotImplemented


# ================================================================
#  Matrix sum, scale, negation and adjoint
# ================================================================
def test_add_and_sub_on_matched_spaces(mx, grid1):
    space = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid1, space)
    a = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    assert jnp.allclose((a + a).data, 2.0 * a.data)
    assert jnp.allclose((a - a).data, 0.0)


def test_add_rejects_mismatched_spaces(mx, my, grid2):
    sa = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    sb = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, sa)
    a = BlockSymbol.from_blocks(((dx,),), (sa,), (sa,))
    b = BlockSymbol.from_blocks(((dx,),), (sa,), (sb,))
    with pytest.raises(SpaceMismatchError, match="mismatched spaces"):
        _ = a + b


def test_add_returns_notimplemented_for_a_foreign_operand(mx, grid1):
    space = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid1, space)
    a = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    assert a.__add__(object()) is NotImplemented
    assert a.__sub__(object()) is NotImplemented


def test_scalar_scale_and_negation(mx, grid1):
    space = mx.fourier(origin=mx.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid1, space)
    a = BlockSymbol.from_blocks(((dx,),), (space,), (space,))
    assert jnp.allclose((a * 2).data, 2.0 * a.data)
    assert jnp.allclose((3 * a).data, 3.0 * a.data)
    assert jnp.allclose((-a).data, -a.data)
    assert a.__mul__(object()) is NotImplemented


def test_conj_is_the_hermitian_adjoint(mx, my, grid2):
    sa = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    sb = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    dx = SpectralDerivative()["x"].eigenvalues(grid2, sa)
    dy = SpectralDerivative()["y"].eigenvalues(grid2, sa)
    a = BlockSymbol.from_blocks(
        ((dx, dy),), (sa, sa), (sb,))
    adj = a.conj()
    # conjugate-transpose of each per-mode matrix + swapped tuples
    assert jnp.allclose(adj.data,
                        jnp.conj(jnp.swapaxes(a.data, -1, -2)))
    assert adj.in_spaces == a.out_spaces
    assert adj.out_spaces == a.in_spaces
