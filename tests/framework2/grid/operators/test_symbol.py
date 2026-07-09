"""Tests for the diagonal ``Symbol`` type and its algebra."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import (
    EigenbasisError,
    Identity,
    ScaledOperator,
)
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.spectral import (
    SpectralDerivative,
    fourier_wavenumbers,
)
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.grid.spaces.constant import ConstantSpace

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
