"""Tests for the per-component ``GridSymbols`` kit."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.symbols import GridSymbols, rayleigh_dual

N2 = 8
N3 = 4


@pytest.fixture
def kit_2d():
    mx = IntervalMesh(N2, (0.0, 1.0), name="x")
    my = IntervalMesh(N2, (0.0, 2.0), name="y")
    grid = Grid((mx, my))
    spaces = {"u": mx.right * my.center,
              "v": mx.center * my.right,
              "p": mx.center * my.center}
    return grid, GridSymbols(grid, spaces), spaces


@pytest.fixture
def kit_3d():
    mx = IntervalMesh(N3, (0.0, 1.0), name="x")
    my = IntervalMesh(N3, (0.0, 1.0), name="y")
    mz = IntervalMesh(N3, (0.0, 1.0), name="z")
    grid = Grid((mx, my, mz))
    spaces = {"u": mx.right * my.center * mz.center,
              "v": mx.center * my.right * mz.center,
              "w": mx.center * my.center * mz.right,
              "p": mx.center * my.center * mz.center}
    return grid, GridSymbols(grid, spaces), spaces


# ================================================================
#  Spaces and transforms
# ================================================================
def test_coeff_is_the_transform_codomain(kit_2d):
    grid, kit, spaces = kit_2d
    bare = spaces["u"].bare
    transform = grid.dispatch.resolve("transform", bare)
    assert kit.coeff("u") is transform.codomain(bare)


def test_forward_and_backward_are_the_bound_directions(kit_2d):
    _, kit, spaces = kit_2d
    fwd = kit.forward("p")
    bwd = kit.backward("p")
    assert fwd.domain is spaces["p"].bare
    assert fwd.codomain is kit.coeff("p")
    assert bwd.domain is kit.coeff("p")
    assert bwd.codomain is spaces["p"].bare


def test_forward_backward_round_trips_a_field(kit_2d):
    grid, kit, spaces = kit_2d
    f = grid.create_field(
        spaces["p"],
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(jnp.pi * y))
    back = kit.backward("p")(kit.forward("p")(f))
    assert back.function_space.bare is spaces["p"].bare
    assert jnp.allclose(back.data, f.data)


# ================================================================
#  Per-axis symbols match the direct operator queries
# ================================================================
def test_diff_matches_the_direct_query(kit_2d):
    grid, kit, _ = kit_2d
    sym = kit.diff("x", on="u")
    direct = FiniteDifference()["x"].eigenvalues(grid, kit.coeff("u"))
    assert sym.space is direct.space  # interned tags
    assert sym.codomain is direct.codomain
    assert jnp.array_equal(sym.data, direct.data)


def test_interp_matches_the_direct_query(kit_3d):
    grid, kit, _ = kit_3d
    sym = kit.interp("z", on="w")
    direct = LinearInterp()["z"].eigenvalues(grid, kit.coeff("w"))
    assert sym.space is direct.space
    assert sym.codomain is direct.codomain
    assert jnp.array_equal(sym.data, direct.data)


# ================================================================
#  move — the composed inter-component staggering
# ================================================================
def test_move_composes_the_differing_axis_interps_2d(kit_2d):
    grid, kit, _ = kit_2d
    moved = kit.move("u", "v")  # differs on x (Right) and y (Center)
    li = LinearInterp()
    cu = kit.coeff("u")
    manual = (li["x"].eigenvalues(grid, cu)
              @ li["y"].eigenvalues(grid, cu))
    assert moved.space is manual.space
    assert moved.codomain is manual.codomain
    assert jnp.array_equal(moved.data, manual.data)


def test_move_composes_a_single_axis_interp_3d(kit_3d):
    grid, kit, _ = kit_3d
    moved = kit.move("u", "p")  # only the x factor differs
    manual = LinearInterp()["x"].eigenvalues(grid, kit.coeff("u"))
    assert moved.space is manual.space
    assert moved.codomain is manual.codomain
    assert jnp.array_equal(moved.data, manual.data)


def test_move_composes_two_of_three_axes_3d(kit_3d):
    grid, kit, _ = kit_3d
    moved = kit.move("u", "w")  # x: Right -> Center, z: Center -> Right
    li = LinearInterp()
    cu = kit.coeff("u")
    manual = (li["x"].eigenvalues(grid, cu)
              @ li["z"].eigenvalues(grid, cu))
    assert moved.space is manual.space
    assert moved.codomain is manual.codomain
    assert jnp.array_equal(moved.data, manual.data)


def test_move_onto_itself_is_the_ones_diagonal(kit_2d):
    _, kit, _ = kit_2d
    one = kit.move("p", "p")
    assert one.codomain is one.space
    assert all(isinstance(f, ConstantSpace) for f in one.space.factors)
    assert jnp.allclose(one.data, 1.0)
    # the ones wildcard is the neutral of ``@``
    sym = kit.diff("x", on="p")
    chain = sym @ one
    assert chain.space is sym.space
    assert chain.codomain is sym.codomain
    assert jnp.array_equal(chain.data, sym.data)


# ================================================================
#  Unknown component names
# ================================================================
def test_unknown_component_raises_a_helpful_key_error(kit_2d):
    _, kit, _ = kit_2d
    with pytest.raises(KeyError, match="unknown component 'q'"):
        kit.coeff("q")
    with pytest.raises(KeyError, match="threads components p, u, v"):
        kit.diff("x", on="rho")
    with pytest.raises(KeyError, match="unknown component"):
        kit.move("u", "nope")
    with pytest.raises(KeyError, match="unknown component"):
        kit.forward("b")
    with pytest.raises(KeyError, match="unknown component"):
        kit.backward("b")


# ================================================================
#  rayleigh_dual — the biorthonormal row
# ================================================================
def test_rayleigh_dual_is_biorthonormal(kit_2d):
    _, kit, _ = kit_2d
    q = {"u": kit.diff("x", on="u"), "v": kit.diff("y", on="v")}
    weights = {"u": 1.0, "v": 1.0}
    p = rayleigh_dual(q, weights)
    norm = sum(weights[c] * q[c].magnitude ** 2 for c in q)
    total = sum(jnp.conj(p[c].data) * q[c].data for c in q)
    mask = norm.data != 0
    assert jnp.allclose(jnp.where(mask, total, 1.0), 1.0)
    # exactly zero on the structural nullspace (kx = ky = 0)
    assert not mask[0, 0]
    assert total[0, 0] == 0.0


def test_rayleigh_dual_weights_scale_the_row(kit_2d):
    _, kit, _ = kit_2d
    q = {"u": kit.diff("x", on="u"), "v": kit.diff("y", on="v")}
    p = rayleigh_dual(q, {"u": 4.0, "v": 1.0})
    total = sum(jnp.conj(p[c].data) * q[c].data for c in q)
    norm = 4.0 * q["u"].magnitude ** 2 + q["v"].magnitude ** 2
    mask = norm.data != 0
    # weighted dual still resolves the identity off the nullspace
    assert jnp.allclose(jnp.where(mask, total, 1.0), 1.0)
    assert total[0, 0] == 0.0
