"""Tests for fridom.framework2.grid.operators.finite_difference."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import EigenbasisError
from fridom.framework2.grid.operators.composed import Laplacian
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.spectral import (
    fourier_wavenumbers,
)
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def fd():
    return FiniteDifference()


# ================================================================
#  Construction and static surface
# ================================================================
def test_order_validation():
    assert FiniteDifference().order == 2
    assert FiniteDifference(order=4).order == 4
    with pytest.raises(ValueError, match="even order"):
        FiniteDifference(order=3)
    with pytest.raises(ValueError, match="even order"):
        FiniteDifference(order=0)


def test_dispatch_kind(fd):
    assert fd.dispatch_kind == "diff"


def test_requirements_halo_is_half_the_order(mx):
    assert FiniteDifference().requirements(mx.center).halo == 1
    assert FiniteDifference(order=4).requirements(mx.center).halo == 2
    assert FiniteDifference().requirements(mx.center).layout == "any"


def test_eigenvalues_is_the_ik_hat_retagging_symbol(fd, mx):
    grid = Grid((mx,))
    sym = fd["x"].eigenvalues(grid, mx.center)
    # retags Fourier(Center) -> Fourier(Right)
    assert sym.space.origin.node_set is NodeSet.CENTER
    assert sym.codomain.origin.node_set is NodeSet.RIGHT
    k = fourier_wavenumbers(mx.fourier(origin=mx.center))
    dx = mx.dx
    expected = (1j * 2.0 * jnp.sin(k * dx / 2.0) / dx
                * jnp.exp(1j * k * 0.5 * dx))
    # the half-cell shift zeroes the even-n Nyquist mode
    assert jnp.allclose(sym.data[:-1], expected[:-1])
    assert sym.data[-1] == 0


def test_bwd_fwd_composes_to_the_real_discrete_laplacian(mx):
    grid = Grid((mx,))
    # div @ grad collapses to the 1x1 block whose entry is bwd @ fwd
    entry = Laplacian().expand(mx.center, grid.dispatch).rows[0][0]
    sym = entry.eigenvalues(grid, mx.center)
    assert sym.space.origin.node_set is NodeSet.CENTER
    assert sym.codomain.origin.node_set is NodeSet.CENTER
    k = fourier_wavenumbers(mx.fourier(origin=mx.center))
    dx = mx.dx
    # real -k_hat**2 = -2 (1 - cos k dx) / dx**2 off the Nyquist mode
    assert jnp.max(jnp.abs(sym.data.imag)) < 1e-12
    khat2 = -2.0 * (1.0 - jnp.cos(k * dx)) / dx ** 2
    assert jnp.allclose(sym.data.real[:-1], khat2[:-1])


def test_eigenvalues_raise_on_the_wrong_boundary(fd, my, mx):
    # bounded meshes diagonalize in the sine/cosine basis, not Fourier
    with pytest.raises(EigenbasisError, match="periodic"):
        fd["y"].eigenvalues(Grid((my,)), my.center)
    # a Fourier operand is not a nodal factor either
    with pytest.raises(EigenbasisError):
        fd.eigenvalues(Grid((mx,)), mx.fourier(origin=mx.center))
    # higher orders are not grounded in iteration 1
    with pytest.raises(EigenbasisError, match="order 2"):
        FiniteDifference(order=4)["x"].eigenvalues(Grid((mx,)),
                                                   mx.center)


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(fd, mx):
    assert fd.codomain(mx.center) is mx.right
    assert fd.codomain(mx.right) is mx.center


def test_codomain_bounded(fd, my):
    assert fd.codomain(my.center) is my.inner
    assert fd.codomain(my.outer) is my.center
    assert fd.codomain(my.inner) is my.center


def test_codomain_preserves_scalars(fd, mx):
    assert fd.codomain(mx.center.as_complex()) is (
        mx.right.as_complex())


def test_codomain_rejects_unlisted_node_sets(fd, mx, my):
    with pytest.raises(SpaceMismatchError, match="no diff signature"):
        fd.codomain(mx.left)
    with pytest.raises(SpaceMismatchError, match="no diff signature"):
        fd.codomain(my.right)  # bounded Right is not in the table


def test_codomain_rejects_average_spaces(fd, mx):
    with pytest.raises(SpaceMismatchError, match="FVDerivative"):
        fd.codomain(mx.cell_avg)


def test_codomain_rejects_bc_structured_spaces(fd, my):
    space = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="BC-free"):
        fd.codomain(space)


# ================================================================
#  Application (window alignment over halo-extended storage)
# ================================================================
def test_periodic_center_to_right_derivative(fd, mx):
    grid = Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    df = fd["x"](f)
    assert df.function_space.bare is mx.right
    x_right = grid.evaluation_nodes(mx.right).data
    exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x_right)
    dx = mx.dx
    assert jnp.allclose(df.data, exact,
                        atol=(2 * jnp.pi) ** 3 * dx ** 2)


def test_periodic_right_to_center_uses_the_wrap_ghost(fd, mx):
    grid = Grid((mx,))
    g = grid.create_field(mx.right, data=jnp.arange(8.0))
    dg = fd["x"](g)
    assert dg.function_space.bare is mx.center
    # interior two-point differences; the first center wraps
    expected = jnp.concatenate(
        [jnp.array([0.0 - 7.0]), jnp.diff(jnp.arange(8.0))]) / mx.dx
    assert jnp.allclose(dg.data, expected)


def test_bounded_center_to_inner_is_exact_on_quadratics(fd, my):
    grid = Grid((my,))
    f = grid.create_field(init=lambda y: y * (2.0 - y))
    df = fd["y"](f)
    assert df.function_space.bare is my.inner
    y_inner = grid.evaluation_nodes(my.inner).data
    assert jnp.allclose(df.data, 2.0 - 2.0 * y_inner)


def test_bounded_outer_to_center(fd, my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y)
    df = fd["y"](f)
    assert df.function_space.bare is my.center
    assert jnp.allclose(df.data, jnp.full(8, 3.0))


def test_bounded_inner_to_center_consumes_the_bc_free_fill(fd, my):
    grid = Grid((my,))
    # du of y(2 - y) lives on inner faces; differencing it back to
    # centers needs the boundary-face ghosts, filled by the one-sided
    # linear extrapolation (exact for the linear du)
    f = grid.create_field(init=lambda y: y * (2.0 - y))
    d2 = fd["y"](fd["y"](f))
    assert d2.function_space.bare is my.center
    assert jnp.allclose(d2.data, jnp.full(8, -2.0))


def test_result_metadata_is_default(fd, mx):
    grid = Grid((mx,))
    f = grid.create_field(name="u", units="m/s")
    assert fd["x"](f).name == "unnamed"  # new quantity


def test_separable_chain_applies_on_multi_axis_fields(fd, mx):
    # regression (wave-3A): the SeparableComposite applies its
    # stored-unbound factors *bound* to the resolved axis, so the
    # staggering kernels' codomain resolution stays unambiguous on
    # >= 2-D operands (operator_algebra_merge.md D5 bound variants)
    my2 = IntervalMesh(8, (0.0, 1.0), name="y")
    grid = Grid((mx, my2))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y))
    chain = (fd @ fd)["x"](f)
    bound_chain = (fd["x"] @ fd["x"])(f)
    sequential = f.diff("x").diff("x")
    assert chain.function_space.bare is (
        sequential.function_space.bare)
    # a mid-chain sync is numerically transparent (T3): the fused
    # chain and the synced sequential path agree bitwise
    assert jnp.array_equal(chain.data, sequential.data)
    assert jnp.array_equal(bound_chain.data, sequential.data)


def test_order_6_needs_a_wider_halo_than_negotiated(mx):
    # provisional halo 2 (the seeded order-2 registry's widest
    # entry is the two-factor FV-derivative chain)
    grid = Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    wide = FiniteDifference(order=6)
    with pytest.raises(ValueError, match="halo width 2"):
        wide["x"](f)


def test_chebyshev_mesh_has_no_fd_signature(fd):
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    with pytest.raises(SpaceMismatchError, match="no center space"):
        fd.codomain(cheb.outer)
