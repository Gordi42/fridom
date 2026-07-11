"""Tests for the ``RealizedMap`` category and its lazy compositions.

The realized-map algebra (S1): the ``RealizedMap`` protocol, the lazy
``RealizedComposite`` / ``RealizedSum`` chains, the ``BoundTransform``
adapter, cross-type ``@`` dispatch (with the eager ``Symbol @ Symbol``
fuse preserved), the explicit-materialization guard, and the
``backward @ symbol.inverse() @ forward`` Poisson solve that proves the
S2 spectral rewrite will compose.
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import Laplacian
from fridom.spatial.operators.realized import (
    BoundTransform,
    RealizedComposite,
    RealizedMap,
    RealizedSum,
    compose_realized,
)
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.operators.symbol import Symbol

TWO_PI = 2.0 * jnp.pi
N = 16


def laplacian_2d():
    return (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
            + SpectralDerivative()["y"] @ SpectralDerivative()["y"])


@pytest.fixture
def setup_2d():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    grid = Grid((mx, my))
    lap = laplacian_2d()
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    nodal = rhs.function_space.bare
    transform = grid.dispatch.resolve("transform", nodal)
    coeff = transform.codomain(nodal)
    sym = lap.eigenvalues(grid, coeff)
    fwd = BoundTransform(transform, nodal)
    bwd = BoundTransform(transform, coeff, backward=True)
    return {
        "grid": grid, "lap": lap, "rhs": rhs, "nodal": nodal,
        "transform": transform, "coeff": coeff, "sym": sym,
        "fwd": fwd, "bwd": bwd,
    }


# ================================================================
#  The protocol membership
# ================================================================
def test_symbol_conforms_to_the_protocol(setup_2d):
    sym = setup_2d["sym"]
    assert isinstance(sym, RealizedMap)
    # ``domain`` is the RealizedMap alias of ``space``
    assert sym.domain is sym.space


def test_realized_maps_conform(setup_2d):
    fwd, bwd, sym = setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"]
    assert isinstance(fwd, RealizedMap)
    assert isinstance(bwd, RealizedMap)
    assert isinstance(sym @ fwd, RealizedMap)
    assert isinstance(fwd + fwd, RealizedMap)


def test_operators_are_not_realized_maps():
    # a static recipe lacks the fixed tags / cheap inverse of a map
    assert not isinstance(Laplacian(), RealizedMap)
    assert not isinstance(SpectralDerivative()["x"], RealizedMap)


# ================================================================
#  Symbol @ Symbol stays the bitwise eager fuse
# ================================================================
def test_symbol_matmul_symbol_fuses_bitwise(setup_2d):
    grid, coeff, lap = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"])
    a = lap.eigenvalues(grid, coeff)
    b = lap.eigenvalues(grid, coeff)
    fused = a @ b
    assert isinstance(fused, Symbol)
    assert jnp.array_equal(fused.data, a.data * b.data)
    assert fused.space is b.space
    assert fused.codomain is a.codomain


# ================================================================
#  Cross-type @ builds a lazy composite
# ================================================================
def test_symbol_at_transform_builds_a_composite(setup_2d):
    sym, fwd = setup_2d["sym"], setup_2d["fwd"]
    chain = sym @ fwd
    assert isinstance(chain, RealizedComposite)
    assert chain.factors == (sym, fwd)
    assert chain.codomain is sym.codomain
    assert chain.domain is fwd.domain


def test_transform_at_symbol_builds_a_composite(setup_2d):
    sym, bwd = setup_2d["sym"], setup_2d["bwd"]
    chain = bwd @ sym
    assert isinstance(chain, RealizedComposite)
    assert chain.factors == (bwd, sym)


def test_composition_flattens_and_is_associative(setup_2d):
    grid, coeff, lap, fwd = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"],
        setup_2d["fwd"])
    a = lap.eigenvalues(grid, coeff)
    b = lap.eigenvalues(grid, coeff)
    left = (a @ b) @ fwd
    right = a @ (b @ fwd)
    # both flatten to two factors: the fused symbol and the transform
    assert isinstance(left, RealizedComposite)
    assert len(left.factors) == len(right.factors) == 2
    assert isinstance(left.factors[0], Symbol)
    assert left.factors[1] is fwd


def test_adjacent_symbols_eager_fuse_inside_a_chain(setup_2d):
    grid, coeff, lap, fwd, bwd = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"],
        setup_2d["fwd"], setup_2d["bwd"])
    a = lap.eigenvalues(grid, coeff)
    b = lap.eigenvalues(grid, coeff)
    # bwd @ a @ b @ fwd -> the two symbols collapse to one factor
    chain = bwd @ a @ b @ fwd
    assert isinstance(chain, RealizedComposite)
    assert len(chain.factors) == 3
    assert isinstance(chain.factors[1], Symbol)
    assert jnp.array_equal(chain.factors[1].data, a.data * b.data)


def test_composite_of_composites_flattens(setup_2d):
    sym, fwd, bwd = setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"]
    left = bwd @ sym          # composite [bwd, sym]
    right = sym @ fwd         # composite [sym, fwd]
    joined = left @ right     # the two syms are adjacent -> fuse
    assert isinstance(joined, RealizedComposite)
    assert len(joined.factors) == 3


def test_compose_realized_collapses_to_a_single_map(setup_2d):
    # a composite whose factors all fuse returns the lone survivor
    grid, coeff, lap = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"])
    a = lap.eigenvalues(grid, coeff)
    b = lap.eigenvalues(grid, coeff)
    fused = compose_realized(a, b)
    assert isinstance(fused, Symbol)


# ================================================================
#  Application, right to left
# ================================================================
def test_composite_applies_right_to_left(setup_2d):
    sym, fwd, bwd, rhs = (
        setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"],
        setup_2d["rhs"])
    chain = bwd @ sym @ fwd
    manual = bwd(sym(fwd(rhs)))
    assert jnp.allclose(chain(rhs).data, manual.data)


def test_backward_symbol_forward_reproduces_poisson(setup_2d):
    # the S2 goal: backward @ symbol.inverse() @ forward == SpectralSolve
    grid, lap, rhs, sym, fwd, bwd = (
        setup_2d["grid"], setup_2d["lap"], setup_2d["rhs"],
        setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"])
    solve = bwd @ sym.inverse() @ fwd
    assert isinstance(solve, RealizedComposite)
    reference = SpectralSolve(lap, grid, rhs.function_space)(rhs)
    assert jnp.allclose(solve(rhs).data, reference.data)
    # and it is the honest mean-free Poisson solution
    assert float(jnp.abs(solve(rhs).mean().data.ravel()[0])) < 1e-13


# ================================================================
#  inverse of a composite (reverse + invert)
# ================================================================
def test_composite_inverse_reverses_and_inverts(setup_2d):
    sym, fwd, bwd, rhs = (
        setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"],
        setup_2d["rhs"])
    solve = bwd @ sym.inverse() @ fwd
    back = solve.inverse()
    assert isinstance(back, RealizedComposite)
    assert len(back.factors) == 3
    # applying inverse then forward recovers the (mean-free) field
    round_trip = solve(back(rhs))
    mean_free = rhs - rhs.mean()
    assert jnp.allclose(round_trip.data, mean_free.data, atol=1e-9)


def test_composite_inverse_threads_where_zero(setup_2d):
    sym, fwd, bwd = setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"]
    solve = bwd @ sym @ fwd
    back = solve.inverse(where_zero=1.0)
    # the inner (inverted) symbol carries the where_zero gauge at k = 0
    inner = back.factors[1]
    assert isinstance(inner, Symbol)
    assert inner.data.ravel()[0] == 1.0


# ================================================================
#  conj of a composite propagates to factors
# ================================================================
def test_composite_conj_conjugates_symbol_factors(setup_2d):
    grid, coeff, lap = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"])
    # a composite of two symbols never exists (they fuse), so build a
    # sum-of-composites is out of scope; conj on a transform-bearing
    # composite raises (the transform conjugate is designed-for)
    a = lap.eigenvalues(grid, coeff)
    fwd = setup_2d["fwd"]
    chain = a @ fwd
    with pytest.raises(NotImplementedError, match="designed-for"):
        chain.conj()


# ================================================================
#  The lazy sum
# ================================================================
def test_sum_applies_and_flattens(setup_2d):
    sym, fwd, bwd, rhs = (
        setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"],
        setup_2d["rhs"])
    left = bwd @ sym @ fwd
    right = bwd @ sym.inverse() @ fwd
    total = left + right
    assert isinstance(total, RealizedSum)
    assert total.domain is left.domain
    assert total.codomain is left.codomain
    manual = left(rhs) + right(rhs)
    assert jnp.allclose(total(rhs).data, manual.data)
    # flatten: (left + right) + left -> three flat terms
    flattened = total + left
    assert isinstance(flattened, RealizedSum)
    assert len(flattened.terms) == 3


def test_sum_radd_and_singleton(setup_2d):
    fwd, bwd, sym, rhs = (
        setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"],
        setup_2d["rhs"])
    a = bwd @ sym @ fwd
    b = bwd @ sym.inverse() @ fwd
    # __radd__ path: a plain map on the left of a composite sum
    total = a.__radd__(b)
    assert isinstance(total, RealizedSum)
    assert jnp.allclose(
        total(rhs).data, (a(rhs) + b(rhs)).data)


def test_sum_has_no_inverse(setup_2d):
    fwd, bwd, sym = setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"]
    total = (bwd @ sym @ fwd) + (bwd @ sym.inverse() @ fwd)
    with pytest.raises(NotImplementedError, match="no closed-form"):
        total.inverse()


def test_sum_matmul_composes(setup_2d):
    grid, coeff, lap, fwd, bwd = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"],
        setup_2d["fwd"], setup_2d["bwd"])
    a = lap.eigenvalues(grid, coeff)
    b = a.inverse()
    # a sum of two coefficient-space symbols, then composed with fwd
    total = (bwd @ a @ fwd) + (bwd @ b @ fwd)
    chain = total @ (fwd.inverse())
    assert isinstance(chain, RealizedComposite)
    assert isinstance(chain.factors[0], RealizedSum)


def test_sum_conj_conjugates_symbol_terms(setup_2d):
    grid, coeff, lap = (
        setup_2d["grid"], setup_2d["coeff"], setup_2d["lap"])
    a = lap.eigenvalues(grid, coeff)
    b = a.inverse()
    # two symbols on the same coeff space would fuse via
    # Symbol.__add__, so build a structural RealizedSum directly:
    total = RealizedSum((a, b))
    conjugated = total.conj()
    assert isinstance(conjugated, RealizedSum)
    assert jnp.array_equal(conjugated.terms[0].data, jnp.conj(a.data))


# ================================================================
#  The bound-transform adapter
# ================================================================
def test_bound_transform_tags_and_direction(setup_2d):
    fwd, bwd, nodal, coeff, transform = (
        setup_2d["fwd"], setup_2d["bwd"], setup_2d["nodal"],
        setup_2d["coeff"], setup_2d["transform"])
    assert fwd.domain is nodal
    assert fwd.codomain is coeff
    assert not fwd.is_backward
    assert fwd.transform is transform
    assert bwd.domain is coeff
    assert bwd.codomain is nodal
    assert bwd.is_backward


def test_bound_transform_call_matches_forward_backward(setup_2d):
    fwd, bwd, transform, rhs = (
        setup_2d["fwd"], setup_2d["bwd"], setup_2d["transform"],
        setup_2d["rhs"])
    rhs_hat = transform.forward(rhs)
    assert jnp.array_equal(fwd(rhs).data, rhs_hat.data)
    assert jnp.array_equal(
        bwd(rhs_hat).data, transform.backward(rhs_hat).data)


def test_bound_transform_inverse_swaps_direction(setup_2d):
    fwd, coeff, nodal = (
        setup_2d["fwd"], setup_2d["coeff"], setup_2d["nodal"])
    inv = fwd.inverse()
    assert inv.is_backward
    assert inv.domain is coeff
    assert inv.codomain is nodal
    # inverse of the inverse is the original direction/tags
    back = inv.inverse()
    assert not back.is_backward
    assert back.domain is nodal
    assert back.codomain is coeff


def test_bound_transform_conj_is_designed_for(setup_2d):
    with pytest.raises(NotImplementedError, match="designed-for"):
        setup_2d["fwd"].conj()


# ================================================================
#  The explicit-materialization guard (decision A)
# ================================================================
def test_symbol_at_operator_raises_materialization_guard(setup_2d):
    sym = setup_2d["sym"]
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = sym @ Laplacian()


def test_operator_at_symbol_raises_materialization_guard(setup_2d):
    sym = setup_2d["sym"]
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = Laplacian() @ sym


def test_composite_at_operator_raises_guard(setup_2d):
    sym, fwd = setup_2d["sym"], setup_2d["fwd"]
    chain = sym @ fwd
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = chain @ Laplacian()
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = Laplacian() @ chain


def test_transform_at_operator_raises_guard(setup_2d):
    fwd = setup_2d["fwd"]
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = fwd @ Laplacian()


def test_operator_left_of_map_raises_via_reflected_dunders(setup_2d):
    # the reflected @ / + on a bound transform and a sum route the
    # left operator through the materialization guard
    fwd, bwd, sym = setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"]
    total = (bwd @ sym @ fwd) + (bwd @ sym.inverse() @ fwd)
    for guarded in (
            lambda: Laplacian() @ fwd,   # BoundTransform.__rmatmul__
            lambda: Laplacian() @ total,  # RealizedSum.__rmatmul__
            lambda: Laplacian() + fwd,   # BoundTransform.__radd__
            lambda: Laplacian() + total):  # RealizedSum.__radd__
        with pytest.raises(SpaceMismatchError,
                           match="materialize it first"):
            guarded()


def test_sum_plus_operator_raises_guard(setup_2d):
    fwd, bwd, sym = setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"]
    total = (bwd @ sym @ fwd) + (bwd @ sym.inverse() @ fwd)
    with pytest.raises(SpaceMismatchError, match="materialize it first"):
        _ = total + Laplacian()


# ================================================================
#  NotImplemented fall-throughs (non-operator, non-map operands)
# ================================================================
def test_matmul_with_a_plain_object_is_not_implemented(setup_2d):
    sym = setup_2d["sym"]
    with pytest.raises(TypeError):
        _ = sym @ object()
    with pytest.raises(TypeError):
        _ = object() @ sym


def test_composite_matmul_with_plain_object_is_not_implemented(setup_2d):
    chain = setup_2d["sym"] @ setup_2d["fwd"]
    with pytest.raises(TypeError):
        _ = chain @ object()
    with pytest.raises(TypeError):
        _ = object() @ chain


def test_sum_with_a_plain_object_is_not_implemented(setup_2d):
    total = ((setup_2d["bwd"] @ setup_2d["sym"] @ setup_2d["fwd"])
             + (setup_2d["bwd"] @ setup_2d["sym"].inverse()
                @ setup_2d["fwd"]))
    with pytest.raises(TypeError):
        _ = total + object()


# ================================================================
#  Pytree registration (grid-bound, jit / scan friendly)
# ================================================================
def test_composite_is_a_registered_pytree(setup_2d):
    sym, fwd, bwd, rhs = (
        setup_2d["sym"], setup_2d["fwd"], setup_2d["bwd"],
        setup_2d["rhs"])
    solve = bwd @ sym.inverse() @ fwd
    leaves, treedef = jax.tree_util.tree_flatten(solve)
    # only the symbol's diagonal is a dynamic leaf; the transforms
    # are static structure
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, RealizedComposite)
    assert jnp.allclose(rebuilt(rhs).data, solve(rhs).data)


def test_sum_is_a_registered_pytree(setup_2d):
    fwd, bwd, sym = setup_2d["fwd"], setup_2d["bwd"], setup_2d["sym"]
    total = (bwd @ sym @ fwd) + (bwd @ sym.inverse() @ fwd)
    leaves, treedef = jax.tree_util.tree_flatten(total)
    assert len(leaves) == 2  # the two symbol diagonals
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, RealizedSum)
