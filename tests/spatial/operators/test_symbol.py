"""Tests for the diagonal ``Symbol`` type and its algebra."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import (
    EigenbasisError,
    Identity,
    ScaledOperator,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.spectral import (
    SpectralDerivative,
    _cosine1_to_sine1,
    _cosine_to_sine,
    _sine1_to_cosine1,
    _sine_to_cosine,
    fourier_wavenumbers,
)
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet

TWO_PI = 2.0 * jnp.pi
N = 16


@pytest.fixture
def periodic():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    return Grid((mx,)), mx


@pytest.fixture
def periodic_2d():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    return Grid((mx, my)), mx, my


@pytest.fixture
def walled():
    mx = IntervalMesh(N, (0.0, 1.0), periodic=False, name="x")
    return Grid((mx,)), mx


@pytest.fixture
def walled_2d():
    mx = IntervalMesh(N, (0.0, 1.0), periodic=False, name="x")
    my = IntervalMesh(N, (0.0, 1.0), periodic=False, name="y")
    return Grid((mx, my)), mx, my


# ----------------------------------------------------------------
#  Trig coefficient spaces (interval length 1.0 in the fixtures)
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


def _d_sine(mesh):
    """d/dx on sine modes: sine k -> cosine k, diagonal pi k."""
    space, cod = _sine2(mesh), _cosine2(mesh)
    k = jnp.arange(cod.shape[0], dtype=jnp.float64)
    return Symbol(space, jnp.pi * k, codomain=cod)


def _d_cosine(mesh):
    """d/dx on cosine modes: cosine k -> sine k, diagonal -pi k."""
    space, cod = _cosine2(mesh), _sine2(mesh)
    k = jnp.arange(1, cod.shape[0] + 1, dtype=jnp.float64)
    return Symbol(space, -jnp.pi * k, codomain=cod)


def _random_coeffs(grid, space, seed):
    rng = np.random.default_rng(seed)
    return grid.create_field(
        space, data=jnp.asarray(rng.standard_normal(space.shape)))


# ================================================================
#  Construction and properties
# ================================================================
def test_construction_stores_bare_tags(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = jnp.arange(space.shape[0], dtype=jnp.complex128)
    sym = Symbol(space, data)
    assert sym.space is space
    assert sym.codomain is space
    assert jnp.array_equal(sym.data, data)


def test_retag_tags_are_independent(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    sym = Symbol(src, jnp.ones(src.shape[0]), codomain=dst)
    assert sym.space is src
    assert sym.codomain is dst


def test_from_field_adopts_the_array_and_space(periodic):
    grid, mx = periodic
    kx = grid.wavenumbers(mx.fourier(origin=mx.center))
    sym = Symbol.from_field(kx)
    assert sym.space is kx.function_space.bare
    assert jnp.array_equal(sym.data, kx.data)


def test_symbol_is_a_registered_pytree(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Symbol(space, jnp.arange(space.shape[0], dtype=jnp.complex128))
    leaves, treedef = jax.tree_util.tree_flatten(sym)
    assert len(leaves) == 1  # only ``_data`` is dynamic
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.space is sym.space
    assert rebuilt.codomain is sym.codomain
    assert jnp.array_equal(rebuilt.data, sym.data)


# ================================================================
#  Application (Hadamard multiply)
# ================================================================
def test_call_is_hadamard_on_the_same_space(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    ft = grid.dispatch.resolve("transform", mx.center)
    f = ft.forward(grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    sym = Symbol(space, 3.0 * jnp.ones(space.shape[0]))
    out = sym(f)
    assert out.function_space is f.function_space
    assert jnp.allclose(out.data, 3.0 * f.data)


def test_call_retag_lands_on_the_codomain(periodic):
    grid, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    ft = grid.dispatch.resolve("transform", mx.center)
    f = ft.forward(grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    sym = Symbol(src, jnp.ones(src.shape[0]), codomain=dst)
    out = sym(f)
    assert out.function_space.bare is dst
    assert jnp.allclose(out.data, f.data)


def test_call_rejects_a_foreign_space(periodic):
    grid, mx = periodic
    other = mx.fourier(origin=mx.right)
    ft = grid.dispatch.resolve("transform", mx.center)
    f = ft.forward(grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    sym = Symbol(other, jnp.ones(other.shape[0]))
    with pytest.raises(SpaceMismatchError, match="applied to a field"):
        sym(f)


# ================================================================
#  Inverse and adjoint
# ================================================================
def test_inverse_regularizes_structural_zeros(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = jnp.arange(space.shape[0], dtype=jnp.float64)  # data[0] == 0
    inv = Symbol(space, data).inverse()
    assert inv.data[0] == 0.0  # the k = 0 nullspace
    assert jnp.allclose(inv.data[1:], 1.0 / data[1:])


def test_inverse_where_zero_and_tag_flip(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    data = jnp.arange(src.shape[0], dtype=jnp.float64)
    inv = Symbol(src, data, codomain=dst).inverse(where_zero=5.0)
    assert inv.space is dst  # tags flip
    assert inv.codomain is src
    assert inv.data[0] == 5.0


def test_conj_conjugates_and_swaps_tags(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    data = jnp.arange(src.shape[0], dtype=jnp.complex128) * 1j
    sym = Symbol(src, data, codomain=dst).conj()
    assert sym.space is dst
    assert sym.codomain is src
    assert jnp.array_equal(sym.data, jnp.conj(data))


# ================================================================
#  Magnitude and elementwise root
# ================================================================
def test_magnitude_collapses_a_retag_onto_the_domain(periodic):
    grid, mx = periodic
    fd = FiniteDifference()["x"].eigenvalues(grid, mx.center)
    assert fd.codomain is not fd.space  # a retagging symbol
    mag = fd.magnitude
    assert mag.space is fd.space
    assert mag.codomain is fd.space  # collapsed: codomain is domain


def test_magnitude_is_the_real_abs_of_the_data(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = (jnp.arange(space.shape[0], dtype=jnp.complex128)
            * (1.0 + 2.0j))
    mag = Symbol(space, data).magnitude
    assert not jnp.iscomplexobj(mag.data)
    assert jnp.allclose(mag.data, jnp.abs(data))


def test_magnitude_squared_matches_conj_matmul(periodic):
    grid, mx = periodic
    s = FiniteDifference()["x"].eigenvalues(grid, mx.center)
    dispersion = (s.conj() @ s).data.real
    assert jnp.allclose(dispersion, s.magnitude.data ** 2)


def test_magnitude_of_the_fd_symbol_is_khat(periodic):
    grid, mx = periodic
    mag = FiniteDifference()["x"].eigenvalues(grid, mx.center).magnitude
    k = fourier_wavenumbers(mx.fourier(origin=mx.center))
    dx = mx.dx
    khat = jnp.abs(2.0 * jnp.sin(k * dx / 2.0) / dx)
    assert jnp.allclose(mag.data.ravel(), khat)


def test_magnitude_keeps_structural_zeros_exact(periodic):
    grid, mx = periodic
    fd = FiniteDifference()["x"].eigenvalues(grid, mx.center)
    assert fd.data.ravel()[0] == 0.0  # the k = 0 structural zero
    assert fd.magnitude.data.ravel()[0] == 0.0  # sqrt(0) == 0 exact


def test_magnitude_squares_compose_on_union_tags(periodic_2d):
    grid, mx, my = periodic_2d
    space = mx.center * my.center
    a = FiniteDifference()["x"].eigenvalues(grid, space)
    b = FiniteDifference()["y"].eigenvalues(grid, space)
    total = a.magnitude ** 2 * b.magnitude ** 2
    # Fourier(x) ⊗ Const(y) unions with Const(x) ⊗ Fourier(y)
    assert total.space.factor("x") is a.space.factor("x")
    assert total.space.factor("y") is b.space.factor("y")
    assert total.codomain is total.space
    assert jnp.allclose(total.data,
                        a.magnitude.data ** 2 * b.magnitude.data ** 2)


def test_sqrt_is_the_elementwise_root(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = jnp.arange(space.shape[0], dtype=jnp.float64)
    rooted = Symbol(space, data).sqrt()
    assert rooted.space is space
    assert rooted.codomain is space
    assert jnp.allclose(rooted.data, jnp.sqrt(data))


def test_sqrt_forbidden_across_a_retag(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    sym = Symbol(src, jnp.ones(src.shape[0]), codomain=dst)
    with pytest.raises(SpaceMismatchError, match="codomain is space"):
        sym.sqrt()


# ================================================================
#  Diagonal algebra
# ================================================================
def test_scalar_arithmetic(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = jnp.arange(space.shape[0], dtype=jnp.float64)
    sym = Symbol(space, data)
    assert jnp.allclose((sym * 2).data, data * 2)
    assert jnp.allclose((2 * sym).data, data * 2)
    assert jnp.allclose((sym + 1).data, data + 1)
    assert jnp.allclose((1 + sym).data, data + 1)
    assert jnp.allclose((sym - 1).data, data - 1)
    assert jnp.allclose((-sym).data, -data)
    assert jnp.allclose((sym / 2).data, data / 2)
    assert jnp.allclose((sym ** 2).data, data ** 2)


def test_scalar_arithmetic_accepts_0d_arrays(periodic):
    """A 0-d jax array is a scalar coefficient (a traced ``1/dsqr``)."""
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    data = jnp.arange(space.shape[0], dtype=jnp.float64)
    sym = Symbol(space, data)
    scalar = jnp.asarray(2.0)  # 0-d, like a traced ctx.params leaf
    assert jnp.allclose((sym * scalar).data, data * 2.0)  # forward
    assert jnp.allclose((scalar * sym).data, data * 2.0)  # reflected
    assert jnp.allclose((sym + scalar).data, data + 2.0)
    assert jnp.allclose((sym / scalar).data, data / 2.0)
    # an n-d array carries no space tags -> not a scalar; must be a field
    nd = jnp.ones(space.shape[0])
    with pytest.raises(TypeError):
        _ = sym * nd
    with pytest.raises(TypeError):
        _ = nd * sym


def test_rtruediv_is_the_tag_flipped_raw_inverse(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    data = jnp.arange(1, src.shape[0] + 1, dtype=jnp.float64)
    inv = 2.0 / Symbol(src, data, codomain=dst)
    assert inv.space is dst
    assert inv.codomain is src
    assert jnp.allclose(inv.data, 2.0 / data)


def test_pow_forbidden_across_a_retag(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    sym = Symbol(src, jnp.ones(src.shape[0]), codomain=dst)
    with pytest.raises(SpaceMismatchError, match="codomain is space"):
        _ = sym ** 2


def test_elementwise_symbol_product_and_sum(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    a = Symbol(space, jnp.arange(space.shape[0], dtype=jnp.float64))
    b = Symbol(space, jnp.ones(space.shape[0]) * 3.0)
    assert jnp.allclose((a * b).data, a.data * 3.0)
    assert jnp.allclose((a + b).data, a.data + 3.0)
    assert jnp.allclose((a / b).data, a.data / 3.0)


def test_matmul_composes_and_threads_the_tags(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    fwd = Symbol(src, jnp.full(src.shape[0], 2.0), codomain=dst)
    bwd = Symbol(dst, jnp.full(dst.shape[0], 5.0), codomain=src)
    chain = bwd @ fwd  # (A @ B) with B.codomain == A.space
    assert chain.space is src
    assert chain.codomain is src
    assert jnp.allclose(chain.data, 10.0)


def test_matmul_rejects_a_tag_mismatch(periodic):
    _, mx = periodic
    src = mx.fourier(origin=mx.center)
    dst = mx.fourier(origin=mx.right)
    a = Symbol(src, jnp.ones(src.shape[0]))
    b = Symbol(dst, jnp.ones(dst.shape[0]))
    with pytest.raises(SpaceMismatchError, match="cannot compose"):
        _ = a @ b


def test_matmul_returns_notimplemented_for_non_symbol(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Symbol(space, jnp.ones(space.shape[0]))
    assert sym.__matmul__(3) is NotImplemented


def test_domain_is_an_alias_of_space(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Symbol(space, jnp.ones(space.shape[0]))
    assert sym.domain is sym.space is space


def test_matmul_with_an_operator_raises_materialization_guard(periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Symbol(space, jnp.ones(space.shape[0]))
    lap = SpectralDerivative()["x"] @ SpectralDerivative()["x"]
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = sym @ lap
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = lap @ sym


def test_elementwise_returns_notimplemented_for_foreign_operand(
        periodic):
    _, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Symbol(space, jnp.ones(space.shape[0]))
    assert sym.__add__(object()) is NotImplemented
    with pytest.raises(TypeError):
        _ = sym + object()


# ================================================================
#  Broadcast (Identity ⊗ D) — the Laplacian kx**2 + ky**2 sum
# ================================================================
def test_broadcast_sum_unions_the_constant_tags(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    ky = SpectralDerivative()["y"].eigenvalues(grid, space)
    total = kx * kx + ky * ky
    # union of Fourier(x) ⊗ Const(y) and Const(x) ⊗ Fourier(y)
    assert total.space is space
    assert total.codomain is space
    kxv = grid.wavenumbers(space, "x").data
    kyv = grid.wavenumbers(space, "y").data
    # off the (annihilated) Nyquist rows/columns the diagonal is the
    # continuous ``-(kx**2 + ky**2)``
    interior = (slice(0, N // 2), slice(0, N // 2))
    expected = -(kxv ** 2) - kyv ** 2
    assert jnp.allclose(total.data[interior], expected[interior])


def test_union_rejects_conflicting_non_constant_factors(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    # a lone Fourier(x) factor vs the x ⊗ y product: rank mismatch
    lone = Symbol(mx.fourier(origin=mx.center),
                  jnp.ones(mx.fourier(origin=mx.center).shape[0]))
    with pytest.raises(SpaceMismatchError, match="incompatible spaces"):
        _ = kx + lone


def test_union_keeps_a_shared_non_constant_factor(periodic_2d):
    _, mx, my = periodic_2d
    full = (mx.fourier(origin=mx.center)
            * my.fourier(origin=my.center))
    x_only = mx.fourier(origin=mx.center) * my.constant
    a = Symbol(full, jnp.ones(full.shape))
    b = Symbol(x_only, jnp.ones(x_only.shape))
    # position x: identical Fourier(x) factor (the ``fa is fb`` path);
    # position y: Fourier(y) unions with Const(y)
    total = a + b
    assert total.space is full


def test_union_rejects_a_disagreeing_non_constant_factor(periodic_2d):
    _, mx, my = periodic_2d
    centered = (mx.fourier(origin=mx.center)
                * my.fourier(origin=my.center))
    staggered = (mx.fourier(origin=mx.right)
                 * my.fourier(origin=my.center))
    a = Symbol(centered, jnp.ones(centered.shape))
    b = Symbol(staggered, jnp.ones(staggered.shape))
    with pytest.raises(SpaceMismatchError, match="non-constant factor"):
        _ = a + b


# ================================================================
#  Symbol x field — the coefficient rule
# ================================================================
def _profile(grid, space, axis, fn):
    """Return a physical profile field on ``axis`` (constant else)."""
    nodes = grid.evaluation_nodes(space, axis)
    return nodes.with_data(fn(nodes.data))


def test_symbol_times_profile_commutes_with_the_transform(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    # a physical profile c(y) — constant on the transformed x axis
    cy = _profile(grid, mx.center * my.center, "y",
                  lambda y: jnp.cos(TWO_PI * y))
    left = kx * cy
    right = cy * kx  # __rmul__ path
    # kx . c(y) commutes: c(y) constant in x passes through FFT_x
    assert jnp.allclose(left.data, right.data)
    # the result adopts the field's Nodal(y) on the symbol's Const(y),
    # keeping Fourier(x) on the transformed axis (mixed representation)
    assert left.space.factor("x") is space.factor("x")
    assert left.space.factor("y") is (mx.center * my.center).factor("y")
    assert jnp.allclose(left.data, kx.data * cy.data)


def test_symbol_times_field_rejects_variation_on_a_transformed_axis(
        periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    # a profile that varies on the transformed x axis is illegal
    cx = _profile(grid, mx.center * my.center, "x",
                  lambda x: jnp.cos(TWO_PI * x))
    with pytest.raises(SpaceMismatchError, match="transformed"):
        _ = kx * cx


def test_symbol_times_constant_field_is_the_dsqr_case(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    # a scalar-ish all-Constant field (the dsqr degenerate case): built
    # by reducing a profile to its (Constant) mean
    const = _profile(grid, mx.center * my.center, "y",
                     lambda y: 0.0 * y + 3.0).mean("y")
    scaled = kx * const
    assert scaled.space is kx.space  # no physical axis introduced
    assert jnp.allclose(scaled.data, 3.0 * kx.data)


def test_symbol_times_field_keeps_a_physical_factor(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    cy = _profile(grid, mx.center * my.center, "y",
                  lambda y: jnp.cos(TWO_PI * y))
    mixed = kx * cy  # Fourier(x) . Center(y)
    # scaling the mixed symbol by an all-Constant field keeps the
    # existing physical Center(y) factor (the ``field Constant`` branch)
    const = _profile(grid, mx.center * my.center, "y",
                     lambda y: 0.0 * y + 2.0).mean("y")
    scaled = mixed * const
    assert scaled.space is mixed.space
    assert jnp.allclose(scaled.data, 2.0 * mixed.data)


def test_symbol_times_field_on_a_matching_profile(periodic_2d):
    grid, mx, my = periodic_2d
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    cy = _profile(grid, mx.center * my.center, "y",
                  lambda y: jnp.cos(TWO_PI * y))
    mixed = kx * cy  # Fourier(x) . Center(y)
    # a second profile on the *same* Center(y) set multiplies in (the
    # ``sym_factor is field_factor`` branch)
    cy2 = _profile(grid, mx.center * my.center, "y",
                   lambda y: 1.0 + jnp.sin(TWO_PI * y))
    both = mixed * cy2
    assert both.space is mixed.space
    assert jnp.allclose(both.data, mixed.data * cy2.data)


def test_symbol_times_field_rejects_a_rank_mismatch(periodic_2d):
    grid, mx, my = periodic_2d
    space = mx.fourier(origin=mx.center)  # 1-factor symbol
    sym = Symbol(space, jnp.ones(space.shape[0]))
    cy = _profile(grid, mx.center * my.center, "y",
                  lambda y: jnp.cos(TWO_PI * y))  # 2-factor field
    with pytest.raises(SpaceMismatchError, match="incompatible ranks"):
        _ = sym * cy


def test_symbol_times_field_rejects_a_disagreeing_profile(periodic_2d):
    grid, mx, my = periodic_2d
    # a symbol already carrying a physical Nodal(y) factor (kx . c(y))
    space = (mx.fourier(origin=mx.center)
             * my.fourier(origin=my.center))
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    cy = _profile(grid, mx.center * my.center, "y",
                  lambda y: jnp.cos(TWO_PI * y))
    mixed = kx * cy  # Fourier(x) . Center(y)
    # multiplying by a profile on a *different* nodal y set disagrees
    staggered = _profile(grid, mx.center * my.right, "y",
                         lambda y: jnp.cos(TWO_PI * y))
    with pytest.raises(SpaceMismatchError, match="physical factor"):
        _ = mixed * staggered


# ================================================================
#  Identity.eigenvalues — the ones diagonal (neutral of ``@``)
# ================================================================
def test_identity_eigenvalues_is_the_ones_diagonal(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    sym = Identity().eigenvalues(grid, space)
    assert isinstance(sym, Symbol)
    # the identity's symbol is the all-Constant ones wildcard
    assert all(isinstance(f, ConstantSpace) for f in sym.space.factors)
    assert sym.codomain is sym.space
    assert jnp.allclose(sym.data, 1.0)


def test_scaled_identity_eigenvalues_is_the_constant(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    csqr = 4.0
    scaled = csqr * Identity()
    sym = scaled.eigenvalues(grid, space)
    assert isinstance(sym, Symbol)
    assert all(isinstance(f, ConstantSpace) for f in sym.space.factors)
    assert sym.codomain is sym.space
    assert jnp.allclose(sym.data, csqr)


def test_scaled_identity_accepts_a_zero_d_array_coeff(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    # a 0-d jax array (a traced-but-constant scalar, e.g. 1/dsqr) is
    # translation-invariant, so the scaled operator keeps a symbol
    scaled = jnp.asarray(2.0) * Identity()
    sym = scaled.eigenvalues(grid, space)
    assert isinstance(sym, Symbol)
    assert all(isinstance(f, ConstantSpace) for f in sym.space.factors)
    assert jnp.allclose(sym.data, 2.0)


def test_scaled_identity_with_a_field_coeff_has_no_symbol(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    # an n-d array carries no space tags -> a field coefficient, which
    # breaks translation invariance: no symbol
    field_coeff = jnp.ones(space.shape)
    scaled = ScaledOperator(field_coeff, Identity())
    with pytest.raises(EigenbasisError, match="field coefficient"):
        scaled.eigenvalues(grid, space)


def test_identity_is_neutral_in_a_symbol_chain(periodic):
    grid, mx = periodic
    space = mx.fourier(origin=mx.center)
    kx = SpectralDerivative()["x"].eigenvalues(grid, space)
    one = Identity().eigenvalues(grid, space)
    chain = kx @ one  # A @ Identity reduces to A
    assert chain.space is kx.space
    assert chain.codomain is kx.codomain
    assert jnp.allclose(chain.data, kx.data)


# ================================================================
#  Cross-axis (disjoint) composition — the Coriolis interp corner
# ================================================================
def test_matmul_composes_disjoint_axes_into_the_corner(periodic_2d):
    grid, mx, my = periodic_2d
    space = mx.center * my.center  # nodal product operand
    # a 2-D interpolation: interp-x ⊗ Const(y) composed against
    # Const(x) ⊗ interp-y — each carries Constant on the other axis
    interp_x = LinearInterp()["x"].eigenvalues(grid, space)
    interp_y = LinearInterp()["y"].eigenvalues(grid, space)
    corner = interp_x @ interp_y
    # domain reads through to the centered product; codomain lands on
    # the fully staggered (Right, Right) corner
    expected_space = (mx.fourier(origin=mx.center)
                      * my.fourier(origin=my.center))
    expected_codomain = (mx.fourier(origin=mx.right)
                         * my.fourier(origin=my.right))
    assert corner.space is expected_space
    assert corner.codomain is expected_codomain
    # the leaf is the outer product of the two per-axis diagonals
    ax = interp_x.data.ravel()
    by = interp_y.data.ravel()
    assert jnp.allclose(corner.data, jnp.outer(ax, by))


def test_matmul_cross_axis_is_order_independent_leaf(periodic_2d):
    grid, mx, my = periodic_2d
    space = mx.center * my.center  # nodal product operand
    interp_x = LinearInterp()["x"].eigenvalues(grid, space)
    interp_y = LinearInterp()["y"].eigenvalues(grid, space)
    # disjoint axes commute at the leaf level (tensor product)
    assert jnp.allclose((interp_x @ interp_y).data,
                        (interp_y @ interp_x).data)


def test_matmul_rejects_a_shared_axis_mismatch_in_2d(periodic_2d):
    _, mx, my = periodic_2d
    # both non-Constant on the shared x axis, disagreeing origins
    a_space = mx.fourier(origin=mx.center) * my.constant
    b_cod = mx.fourier(origin=mx.right) * my.constant
    a = Symbol(a_space, jnp.ones(a_space.shape))
    b = Symbol(mx.fourier(origin=mx.center) * my.constant,
               jnp.ones(a_space.shape), codomain=b_cod)
    with pytest.raises(SpaceMismatchError, match="cannot compose"):
        _ = a @ b


# ================================================================
#  Derived shifts — sine/cosine slot layouts differ by one mode
# ================================================================
def test_shifted_apply_sine2_matches_the_kernel(walled):
    grid, mx = walled
    sym = _d_sine(mx)
    f = _random_coeffs(grid, _sine2(mx), seed=1)
    out = sym(f)
    assert out.function_space.bare is _cosine2(mx)
    expected = _sine_to_cosine(jnp.asarray(f.data), 0, 1.0)
    assert jnp.array_equal(out.data, expected)
    assert out.data[0] == 0.0  # cosine k = 0: zero-filled


def test_shifted_apply_cosine2_matches_the_kernel(walled):
    grid, mx = walled
    sym = _d_cosine(mx)
    f = _random_coeffs(grid, _cosine2(mx), seed=2)
    out = sym(f)
    assert out.function_space.bare is _sine2(mx)
    expected = _cosine_to_sine(jnp.asarray(f.data), 0, 1.0)
    assert jnp.array_equal(out.data, expected)
    assert out.data[-1] == 0.0  # sine k = n: zero-filled


def test_shifted_apply_sine1_matches_the_kernel(walled):
    grid, mx = walled
    space, cod = _sine1(mx), _cosine1(mx)  # n - 1 -> n + 1 slots
    k = jnp.arange(cod.shape[0], dtype=jnp.float64)
    sym = Symbol(space, jnp.pi * k, codomain=cod)
    f = _random_coeffs(grid, space, seed=3)
    out = sym(f)
    assert out.function_space.bare is cod
    assert out.shape == (N + 1,)
    expected = _sine1_to_cosine1(jnp.asarray(f.data), 0, 1.0)
    assert jnp.array_equal(out.data, expected)
    # neither cosine k = 0 nor the Nyquist k = n is populated
    assert out.data[0] == 0.0
    assert out.data[-1] == 0.0


def test_shifted_apply_cosine1_matches_the_kernel(walled):
    grid, mx = walled
    space, cod = _cosine1(mx), _sine1(mx)  # n + 1 -> n - 1 slots
    k = jnp.arange(1, cod.shape[0] + 1, dtype=jnp.float64)
    sym = Symbol(space, -jnp.pi * k, codomain=cod)
    f = _random_coeffs(grid, space, seed=4)
    out = sym(f)
    assert out.function_space.bare is cod
    assert out.shape == (N - 1,)
    expected = _cosine1_to_sine1(jnp.asarray(f.data), 0, 1.0)
    assert jnp.array_equal(out.data, expected)


def test_shifted_matmul_fuses_the_sine_laplacian(walled):
    grid, mx = walled
    fwd = _d_sine(mx)  # Sine -> Cosine
    bwd = _d_cosine(mx)  # Cosine -> Sine
    lap = bwd @ fwd
    assert lap.space is _sine2(mx)
    assert lap.codomain is _sine2(mx)
    # hand-built index map: the inner cosine-layout diagonal shifts
    # by -1 into the sine layout (drop k = 0, zero-fill k = n)
    inner = jnp.concatenate([fwd.data[1:], jnp.zeros(1)])
    assert jnp.array_equal(lap.data, bwd.data * inner)
    assert lap.data[-1] == 0.0  # intermediate-only mode annihilated
    # fused apply == chained apply == chained kernels
    f = _random_coeffs(grid, _sine2(mx), seed=5)
    chained = bwd(fwd(f))
    assert jnp.allclose(lap(f).data, chained.data)
    kernels = _cosine_to_sine(
        _sine_to_cosine(jnp.asarray(f.data), 0, 1.0), 0, 1.0)
    assert jnp.allclose(lap(f).data, kernels)


def test_shifted_magnitude_reindexes_to_the_domain_layout(walled):
    _, mx = walled
    mag = _d_sine(mx).magnitude
    assert mag.space is _sine2(mx)
    assert mag.codomain is _sine2(mx)
    # sine slot j holds mode j + 1: |pi (j + 1)|, top mode dropped
    k = jnp.arange(1, N, dtype=jnp.float64)
    assert jnp.array_equal(mag.data[:-1], jnp.pi * k)
    assert mag.data[-1] == 0.0  # mode n has no cosine image


def test_shifted_conj_lands_in_the_swapped_codomain_layout(walled):
    _, mx = walled
    sym = _d_sine(mx)
    adj = sym.conj()
    assert adj.space is _cosine2(mx)
    assert adj.codomain is _sine2(mx)
    # the adjoint data lives in the sine layout: slot j holds the
    # conjugate eigenvalue of mode j + 1; mode n is zero-filled
    k = jnp.arange(1, N, dtype=jnp.float64)
    assert jnp.array_equal(adj.data[:-1], jnp.pi * k)
    assert adj.data[-1] == 0.0


def test_shifted_conj_round_trips(walled):
    _, mx = walled
    sym = _d_sine(mx)
    back = sym.conj().conj()
    assert back.space is sym.space
    assert back.codomain is sym.codomain
    assert jnp.array_equal(back.data, sym.data)


def test_inverse_guard_raises_across_a_derived_shift(walled):
    _, mx = walled
    with pytest.raises(SpaceMismatchError,
                       match="no diagonal inverse"):
        _d_sine(mx).inverse()


def test_inverse_works_at_net_shift_zero(walled):
    _, mx = walled
    lap = _d_cosine(mx) @ _d_sine(mx)  # sine -> sine: net shift 0
    inv = lap.inverse()
    assert inv.space is lap.codomain
    assert inv.codomain is lap.space
    assert inv.data[-1] == 0.0  # the annihilated top mode: regular
    assert jnp.allclose(inv.data[:-1], 1.0 / lap.data[:-1])


def test_mode_embedding_pads_the_tail(walled):
    grid, mx = walled
    # equal offsets, different lengths: DCT-II modes 0..n-1 embed
    # into DCT-I modes 0..n (a zero-pad at the absent Nyquist)
    space, cod = _cosine2(mx), _cosine1(mx)
    sym = Symbol(space, jnp.ones(cod.shape[0]), codomain=cod)
    f = _random_coeffs(grid, space, seed=6)
    out = sym(f)
    assert out.function_space.bare is cod
    assert jnp.array_equal(out.data[:-1], f.data)
    assert out.data[-1] == 0.0


# ================================================================
#  Derived shifts on a product — trailing-axis alignment
# ================================================================
def test_shifted_apply_moves_only_the_shifted_axis(walled_2d):
    grid, mx, my = walled_2d
    space = _sine2(mx) * _sine2(my)
    cod = _cosine2(mx) * _sine2(my)
    k = jnp.arange(N, dtype=jnp.float64)
    sym = Symbol(space, (jnp.pi * k)[:, None], codomain=cod)
    f = _random_coeffs(grid, space, seed=7)
    out = sym(f)
    assert out.function_space.bare is cod
    expected = _sine_to_cosine(jnp.asarray(f.data), 0, 1.0)
    assert jnp.array_equal(out.data, expected)


def test_shifted_matmul_skips_a_broadcast_inner_axis(walled_2d):
    _, mx, my = walled_2d
    # the outer shifts along x; the inner is Identity ⊗ D along y,
    # so its data is constant on every x mode and must pass through
    outer_space = _cosine2(mx) * my.constant
    outer_cod = _sine2(mx) * my.constant
    k = jnp.arange(1, N + 1, dtype=jnp.float64)
    outer = Symbol(outer_space, (-jnp.pi * k)[:, None],
                   codomain=outer_cod)
    inner_space = mx.constant * _sine2(my)
    diag_y = jnp.arange(1.0, N + 1.0)
    # 1-D data: trailing-aligned to y, no x axis at all
    inner_1d = Symbol(inner_space, diag_y)
    fused = outer @ inner_1d
    assert fused.space is _cosine2(mx) * _sine2(my)
    assert fused.codomain is _sine2(mx) * _sine2(my)
    assert jnp.array_equal(fused.data, outer.data * inner_1d.data)
    # size-1 x axis: same passthrough
    inner_2d = Symbol(inner_space, diag_y[None, :])
    assert jnp.array_equal((outer @ inner_2d).data,
                           outer.data * inner_2d.data)


# ================================================================
#  The shiftless path is bitwise-identical to the plain Hadamard
# ================================================================
def test_shiftless_paths_stay_bitwise_identical(periodic):
    grid, mx = periodic
    # a retagging Fourier symbol: equal (mode_offset, shape) on the
    # center/right origins, so no derived shift anywhere
    fd = FiniteDifference()["x"].eigenvalues(grid, mx.center)
    ft = grid.dispatch.resolve("transform", mx.center)
    f = ft.forward(grid.create_field(
        init=lambda x: jnp.sin(TWO_PI * x)))
    # apply: the plain Hadamard product, bit for bit
    assert jnp.array_equal(fd(f).data, f.data * fd.data)
    # fuse: the plain diagonal product, bit for bit
    bwd = fd.conj()
    assert jnp.array_equal((bwd @ fd).data, bwd.data * fd.data)
    # conj / magnitude: the plain elementwise forms, bit for bit
    assert jnp.array_equal(bwd.data, jnp.conj(fd.data))
    assert jnp.array_equal(
        fd.magnitude.data,
        jnp.sqrt(jnp.real(jnp.conj(fd.data) * fd.data)))
