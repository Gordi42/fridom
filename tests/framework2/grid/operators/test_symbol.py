"""Tests for the diagonal ``Symbol`` type and its algebra."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import Identity
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.spectral import SpectralDerivative
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
#  Concrete probe: every resolved linear block carries a symbol
# ================================================================
def _nonhydro_model():
    import fridom.nonhydro2 as nh  # noqa: PLC0415 — heavy model package
    grid = Grid(tuple(
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name=name)
        for name in ("x", "y", "z")))
    return nh.Model(grid=grid, dt=0.02)


def _shallowwater_model():
    import fridom.framework2 as fr  # noqa: PLC0415
    import fridom.shallowwater2 as sw  # noqa: PLC0415 — heavy package
    grid = Grid(tuple(
        IntervalMesh(16, (0.0, 1.0), periodic=True, name=name)
        for name in ("x", "y")))
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.mark.parametrize(
    "build", [_nonhydro_model, _shallowwater_model],
    ids=["nonhydro", "shallowwater"])
def test_every_linear_block_resolves_to_a_symbol(build):
    import fridom.framework2 as fr  # noqa: PLC0415
    model = build()
    blocks = fr.linear_blocks(model)
    assert blocks  # the model exposes linear terms to diagonalize
    for block in blocks:
        src_space = model.state[block.src].function_space
        sym = block.op.eigenvalues(model.grid, src_space)
        assert isinstance(sym, Symbol)
