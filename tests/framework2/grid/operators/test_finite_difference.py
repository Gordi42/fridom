"""Tests for fridom.framework2.grid.operators.finite_difference."""
import jax.numpy as jnp
import numpy as np
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
from fridom.framework2.grid.operators.trig import Cosine, Sine
from fridom.framework2.grid.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(N, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def mz():
    return IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")


@pytest.fixture
def walled(mz):
    return Grid((mz,))


@pytest.fixture
def fd():
    return FiniteDifference()


def _mode_field(grid, mesh, space, family, index):
    """Synthesize one trig mode through the backward transform."""
    coeff_space = (mesh.sine(space) if family is Sine
                   else mesh.cosine(space))
    data = jnp.zeros(coeff_space.shape[0]).at[index].set(1.0)
    coeff = grid.create_field(coeff_space, data=data)
    return family(grid).backward(coeff)


# ----------------------------------------------------------------
#  Trig coefficient spaces of a walled mesh
# ----------------------------------------------------------------
def _sine2(mesh):
    """DST-II space: modes 1..n at slots 0..n-1."""
    return mesh.sine(mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))


def _sine1(mesh):
    """DST-I space: modes 1..n-1 at slots 0..n-2."""
    return mesh.sine(mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))


def _cosine2(mesh):
    """DCT-II space: modes 0..n-1 at slots 0..n-1."""
    return mesh.cosine(mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN))


def _cosine1(mesh):
    """DCT-I space: modes 0..n at slots 0..n."""
    return mesh.cosine(mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN))


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
    # the Nyquist leaf is kept: 2i sin(π/2)/dx · e^{iπ/2} = -2/dx (real,
    # representable — matching the staggered_diff kernel), unlike a pure
    # phase shift where a half-cell shift of the real Nyquist is zeroed
    assert jnp.allclose(sym.data, expected)
    # the Nyquist entry is snapped to its exact analytic value: the
    # ~1e-16 spurious imaginary part of the exp/sin round trip is gone
    assert sym.data.ravel()[-1] == -2.0 / dx
    assert sym.data.ravel()[-1].imag == 0.0


def test_bwd_fwd_composes_to_the_real_discrete_laplacian(mx):
    grid = Grid((mx,))
    # div @ grad collapses to the 1x1 block whose entry is bwd @ fwd
    entry = Laplacian().expand(mx.center, grid.dispatch).rows[0][0]
    sym = entry.eigenvalues(grid, mx.center)
    assert sym.space.origin.node_set is NodeSet.CENTER
    assert sym.codomain.origin.node_set is NodeSet.CENTER
    k = fourier_wavenumbers(mx.fourier(origin=mx.center))
    dx = mx.dx
    # real -k_hat**2 = -2 (1 - cos k dx) / dx**2 on every mode — the
    # bwd @ fwd round-trip recovers the Nyquist (-4/dx**2), matching the
    # discrete div @ grad kernel exactly (the pressure Poisson symbol)
    assert jnp.max(jnp.abs(sym.data.imag)) < 1e-12
    khat2 = -2.0 * (1.0 - jnp.cos(k * dx)) / dx ** 2
    assert jnp.allclose(sym.data.real, khat2)


def test_eigenvalues_raise_on_the_wrong_boundary(fd, my, mx):
    # bounded meshes diagonalize in the sine/cosine basis, not Fourier
    with pytest.raises(EigenbasisError, match="periodic"):
        fd["y"].eigenvalues(Grid((my,)), my.center)
    # higher orders are not grounded in iteration 1
    with pytest.raises(EigenbasisError, match="order 2"):
        FiniteDifference(order=4)["x"].eigenvalues(Grid((mx,)),
                                                   mx.center)


def test_eigenvalues_thread_a_fourier_coefficient_factor(fd, mx):
    # layout-faithful threading (decision 3): a periodic Fourier factor
    # is a legal operand and yields the same i k_hat symbol as the
    # nodal query (the origin fixes the staggering, its scalars fix the
    # spectrum layout)
    grid = Grid((mx,))
    nodal = fd.eigenvalues(grid, mx.center)
    coeff = fd.eigenvalues(grid, mx.fourier(origin=mx.center))
    assert coeff.space is nodal.space
    assert coeff.codomain is nodal.codomain
    assert jnp.array_equal(coeff.data, nodal.data)


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(fd, mx):
    assert fd.codomain(mx.center) is mx.right
    assert fd.codomain(mx.right) is mx.center


def test_codomain_retags_a_fourier_factor(fd, mx):
    # layout-faithful eigenvalue threading (decision 3): the codomain
    # of a Fourier factor retags through its staggered origin
    src = mx.fourier(origin=mx.center)
    assert fd.codomain(src) is mx.fourier(origin=mx.right)


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


def test_codomain_bc_tagged_maps_to_the_bc_free_sibling(fd, my):
    # the BC tag governs only the ghost fill; nodal outputs are
    # BC-free (owner decision)
    assert fd.codomain(
        my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)) is my.inner
    assert fd.codomain(
        my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)) is my.inner
    assert fd.codomain(
        my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)) is my.center
    assert fd.codomain(
        my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)) is my.center


def test_codomain_rejects_dirichlet_dropped_membership(fd, my):
    # Dirichlet on a member node set drops the boundary value DOF;
    # the staggered stencils cannot align on such lattices
    with pytest.raises(SpaceMismatchError, match="boundary DOF"):
        fd.codomain(my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))
    with pytest.raises(SpaceMismatchError, match="boundary DOF"):
        fd.codomain(my.nodal(NodeSet.RIGHT, bc=BC.DIRICHLET))
    with pytest.raises(SpaceMismatchError, match="boundary DOF"):
        fd.codomain(my.nodal(
            NodeSet.OUTER, bc=(BC.DIRICHLET, BC.NEUMANN)))


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


# ================================================================
#  BC-tagged bounded domains (C3: the tag governs the ghost fill)
# ================================================================
@pytest.mark.parametrize("k0", [1, 3, N], ids=["first", "mid", "top"])
def test_dirichlet_center_mode_derivative(fd, walled, mz, k0):
    # sin(k0 pi z) at centers (DST-II synthesis) -> k_hat cos at the
    # interior faces: the discrete staggered derivative is exact on
    # single modes, k_hat = 2 sin(k dz/2) / dz
    space = mz.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    f = _mode_field(walled, mz, space, Sine, k0 - 1)
    df = fd["z"](f)
    assert df.function_space.bare is mz.inner  # BC-free sibling
    k = k0 * jnp.pi
    k_hat = 2.0 * jnp.sin(k * mz.dx / 2.0) / mz.dx
    z = walled.evaluation_nodes(mz.inner).data
    assert jnp.allclose(df.data, k_hat * jnp.cos(k * z), atol=1e-14)


def test_neumann_center_mode_derivative(fd, walled, mz):
    # d/dz of cos(k z) sampled at centers -> -k_hat times the
    # half-shifted sine at the interior faces
    space = mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    k0 = 3
    f = _mode_field(walled, mz, space, Cosine, k0)
    df = fd["z"](f)
    assert df.function_space.bare is mz.inner
    k = k0 * jnp.pi
    k_hat = 2.0 * jnp.sin(k * mz.dx / 2.0) / mz.dx
    z = walled.evaluation_nodes(mz.inner).data
    assert jnp.allclose(df.data, -k_hat * jnp.sin(k * z), atol=1e-14)


def test_dirichlet_inner_mode_derivative(fd, walled, mz):
    # sin(k0 pi z) on the interior faces (DST-I synthesis) -> k_hat
    # cos at the centers; the wall rows consume the Dirichlet w = 0
    # ghost, which coincides with the analytic sin(0) = sin(k L) = 0
    space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    k0 = 3
    f = _mode_field(walled, mz, space, Sine, k0 - 1)
    df = fd["z"](f)
    assert df.function_space.bare is mz.center
    k = k0 * jnp.pi
    k_hat = 2.0 * jnp.sin(k * mz.dx / 2.0) / mz.dx
    z = walled.evaluation_nodes(mz.center).data
    assert jnp.allclose(df.data, k_hat * jnp.cos(k * z), atol=1e-14)


def test_neumann_outer_mode_derivative(fd, walled, mz):
    # cos(k0 pi z) on all faces (DCT-I synthesis) -> -k_hat sin at
    # the centers (no ghost consumption: centers sit between faces)
    space = mz.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    k0 = 3
    f = _mode_field(walled, mz, space, Cosine, k0)
    df = fd["z"](f)
    assert df.function_space.bare is mz.center
    k = k0 * jnp.pi
    k_hat = 2.0 * jnp.sin(k * mz.dx / 2.0) / mz.dx
    z = walled.evaluation_nodes(mz.center).data
    assert jnp.allclose(df.data, -k_hat * jnp.sin(k * z), atol=1e-14)


def test_dirichlet_inner_wall_rows_use_zero_ghosts(fd, walled, mz):
    # the w Dirichlet fill: the vacant wall faces are zero-value
    # ghosts (tensor.py Inner-Dirichlet face lattice); the wall rows
    # of the derivative match a manual stencil with w = 0 ghosts
    space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    w = jnp.asarray([2.0, -1.0, 4.0, 0.5, -3.0, 1.5, 2.5])
    f = walled.create_field(space, data=w)
    df = fd["z"](f)
    w_ext = jnp.concatenate([jnp.zeros(1), w, jnp.zeros(1)])
    assert jnp.allclose(df.data, jnp.diff(w_ext) / mz.dx)


def test_bc_tagged_diff_is_registry_resolvable(walled, mz):
    # a walled grid seeds ("diff", tagged-origin) rows, so the field
    # verb dispatches without manual seeding
    space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    fd = walled.dispatch.resolve("diff", space)
    assert isinstance(fd, FiniteDifference)
    assert fd is walled.dispatch.resolve("diff", mz.center)
    w = jnp.arange(1.0, 8.0)
    f = walled.create_field(space, data=w)
    df = f.diff("z")
    assert df.function_space.bare is mz.center
    w_ext = jnp.concatenate([jnp.zeros(1), w, jnp.zeros(1)])
    assert jnp.allclose(df.data, jnp.diff(w_ext) / mz.dx)


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


# ================================================================
#  Trig eigenvalue symbols (C4: bounded staggering diagonals)
# ================================================================
def test_trig_codomain_pairing_flips_family_and_bc(fd, mz):
    # coefficient-side codomains carry constitutive BC tags (the
    # nodal outputs above stay BC-free — two-representations rule):
    # diff flips family and BC kind, staggering the node set
    assert fd.codomain(_sine2(mz)) is _cosine1(mz)
    assert fd.codomain(_sine1(mz)) is _cosine2(mz)
    assert fd.codomain(_cosine2(mz)) is _sine1(mz)
    assert fd.codomain(_cosine1(mz)) is _sine2(mz)


def test_trig_codomain_rejects_unlisted_origins(fd, mz):
    # a sine space of a Left-Dirichlet origin is off the four-family
    # staggering table
    left = mz.sine(mz.nodal(NodeSet.LEFT, bc=BC.DIRICHLET))
    with pytest.raises(SpaceMismatchError, match="no diff pairing"):
        fd.codomain(left)


@pytest.mark.parametrize(
    ("domain_of", "family_in", "family_out", "trim_walls"),
    [pytest.param(_sine2, Sine, Cosine, True, id="sine2->cosine1"),
     pytest.param(_cosine2, Cosine, Sine, False, id="cosine2->sine1"),
     pytest.param(_sine1, Sine, Cosine, False, id="sine1->cosine2"),
     pytest.param(_cosine1, Cosine, Sine, False, id="cosine1->sine2")])
def test_trig_symbol_matches_the_nodal_apply(
        fd, walled, mz, domain_of, family_in, family_out, trim_walls):
    # the decisive identity on random fields:
    # backward(symbol(forward(f))) == BC-aware nodal derivative.
    # The sine2 -> cosine1 codomain extends to the wall faces; its
    # interior slice is the nodal Inner output
    domain = domain_of(mz)
    rng = np.random.default_rng(3)
    f = walled.create_field(
        domain.origin,
        data=jnp.asarray(rng.standard_normal(domain.origin.shape)))
    sym = fd["z"].eigenvalues(walled, domain)
    back = family_out(walled).backward(
        sym(family_in(walled).forward(f)))
    ref = fd["z"](f)
    out = back.data[1:-1] if trim_walls else back.data
    assert jnp.max(jnp.abs(out - ref.data)) < 1e-14


def test_trig_eigenvalues_match_the_periodic_magnitude_table(
        fd, walled, mz):
    dz = mz.dx
    # sine -> cosine carries +k_hat = 2 sin(k dz/2)/dz — the same
    # magnitude as the periodic table at k = pi m / L, evaluated on
    # the codomain (cosine-I) mode table; real: walls kill the phase
    plus = fd["z"].eigenvalues(walled, _sine2(mz))
    k = jnp.pi * jnp.arange(N + 1)  # cosine-I modes 0..n
    assert jnp.array_equal(plus.data.ravel(),
                           2.0 * jnp.sin(k * dz / 2.0) / dz)
    assert not jnp.iscomplexobj(plus.data)
    # cosine -> sine carries -k_hat (the _cosine_to_sine sign)
    minus = fd["z"].eigenvalues(walled, _cosine1(mz))
    k = jnp.pi * jnp.arange(1, N + 1)  # sine-II modes 1..n
    assert jnp.array_equal(minus.data.ravel(),
                           -(2.0 * jnp.sin(k * dz / 2.0) / dz))


def test_trig_eigenvalues_structural_zero_at_cosine_mode_zero(
        fd, walled, mz):
    # diff-from-sine never populates cosine k = 0: the entry is an
    # exact structural zero (it multiplies the embedding zero-fill)
    s2 = fd["z"].eigenvalues(walled, _sine2(mz))
    assert s2.data.ravel()[0] == 0.0
    s1 = fd["z"].eigenvalues(walled, _sine1(mz))
    assert s1.data.ravel()[0] == 0.0


@pytest.mark.parametrize("domain_of",
                         [_sine2, _sine1, _cosine2, _cosine1])
def test_trig_eigenvalues_thread_the_tagged_nodal_factor(
        fd, walled, mz, domain_of):
    # layout-faithful threading: the BC-tagged nodal origin resolves
    # the same symbol as the trig coefficient factor
    coeff_space = domain_of(mz)
    nodal = fd["z"].eigenvalues(walled, coeff_space.origin)
    coeff = fd["z"].eigenvalues(walled, coeff_space)
    assert nodal.space is coeff.space
    assert nodal.codomain is coeff.codomain
    assert jnp.array_equal(nodal.data, coeff.data)
