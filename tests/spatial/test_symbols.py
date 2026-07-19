"""Tests for the per-component ``GridSymbols`` kit."""
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.mixed import (
    ComposedTransform,
    resolve_transform,
)
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.symbols import (
    GridSymbols,
    ModeChart,
    rayleigh_dual,
)

N2 = 8
N3 = 4
NW = 8


@pytest.fixture
def kit_2d():
    mx = IntervalMesh(N2, (0.0, 1.0), name="x")
    my = IntervalMesh(N2, (0.0, 2.0), name="y")
    grid = Grid((mx, my), device_ids=(0,))
    spaces = {"u": mx.right * my.center,
              "v": mx.center * my.right,
              "p": mx.center * my.center}
    return grid, GridSymbols(grid, spaces), spaces


@pytest.fixture
def kit_3d():
    mx = IntervalMesh(N3, (0.0, 1.0), name="x")
    my = IntervalMesh(N3, (0.0, 1.0), name="y")
    mz = IntervalMesh(N3, (0.0, 1.0), name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    spaces = {"u": mx.right * my.center * mz.center,
              "v": mx.center * my.right * mz.center,
              "w": mx.center * my.center * mz.right,
              "p": mx.center * my.center * mz.center}
    return grid, GridSymbols(grid, spaces), spaces


@pytest.fixture
def kit_walled():
    # the walled-z kit: periodic x/y, bounded z with BC-tagged
    # component factors (trig transforms reject BC-free origins)
    mx = IntervalMesh(NW, (0.0, 1.0), name="x")
    my = IntervalMesh(NW, (0.0, 2.0), name="y")
    mz = IntervalMesh(NW, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    hor = mx.center * my.center
    spaces = {
        "w": hor * mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
        "b": hor * mz.nodal(NodeSet.CENTER, bc=BC.DIRICHLET),
        "p": hor * mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN),
    }
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
#  Walled grids: mixed per-component transforms (C5)
# ================================================================
# NOTE: kit.diff / kit.interp on the walled grid need the trig
# eigenvalue rows (the parallel C4 work); they are exercised with
# the C7 pressure-solve integration.
@pytest.mark.parametrize(("name", "z_family"), [
    pytest.param("w", SineSpace, id="w-inner-dirichlet"),
    pytest.param("b", SineSpace, id="b-center-dirichlet"),
    pytest.param("p", CosineSpace, id="p-center-neumann"),
])
def test_walled_coeff_is_the_mixed_product(kit_walled, name,
                                           z_family):
    grid, kit, spaces = kit_walled
    coeff = kit.coeff(name)
    assert isinstance(coeff.factor("x"), FourierSpace)
    assert isinstance(coeff.factor("y"), FourierSpace)
    assert isinstance(coeff.factor("z"), z_family)
    # interned identity against the resolved composition
    tf = resolve_transform(grid, spaces[name].bare)
    assert isinstance(tf, ComposedTransform)
    assert coeff is tf.codomain(spaces[name].bare)


@pytest.mark.parametrize("name", ["w", "b", "p"])
def test_walled_forward_backward_round_trips(kit_walled, name):
    grid, kit, spaces = kit_walled
    f = grid.random.normal(spaces[name], seed=21)
    fwd = kit.forward(name)
    bwd = kit.backward(name)
    assert fwd.domain is spaces[name].bare
    assert fwd.codomain is kit.coeff(name)
    assert bwd.domain is kit.coeff(name)
    assert bwd.codomain is spaces[name].bare
    back = bwd(fwd(f))
    assert back.function_space is f.function_space
    assert jnp.allclose(back.data, f.data, atol=1e-14)


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
#  ModeChart — the union mode lattice of the trig families
# ================================================================
@pytest.mark.parametrize(("name", "offset"), [
    pytest.param("w", 1, id="sine1-modes-1-to-n-1"),
    pytest.param("b", 1, id="sine2-modes-1-to-n"),
    pytest.param("p", 0, id="cosine2-modes-0-to-n-1"),
])
def test_mode_chart_embeds_and_restricts_the_trig_lattices(
        kit_walled, name, offset):
    grid, kit, _ = kit_walled
    chart = ModeChart(grid)
    coeff = kit.coeff(name)
    slots = coeff.factor("z").shape[0]
    data = jnp.arange(1.0, slots + 1).reshape(1, 1, slots)
    up = chart.embed(data, coeff)
    # slot j (mode j + offset) lands on union slot j + offset ...
    assert up.shape[-1] == NW + 1
    assert jnp.array_equal(up[..., offset:offset + slots], data)
    # ... and the modes the component lacks are exact zeros
    assert jnp.all(up[..., :offset] == 0.0)
    assert jnp.all(up[..., offset + slots:] == 0.0)
    # restrict inverts the embedding
    assert jnp.array_equal(chart.restrict(up, coeff), data)


def test_mode_chart_is_identity_on_the_dct1_lattice(kit_walled):
    # the DCT-I (Neumann Outer) family already IS the 0..n union
    # lattice, so the chart passes it through untouched
    grid, _, _ = kit_walled
    mz = next(m for m in grid.factors if "z" in m.names)
    cos1 = mz.cosine(mz.nodal(NodeSet.OUTER, bc=BC.NEUMANN))
    chart = ModeChart(grid)
    data = jnp.arange(1.0, NW + 2)
    assert chart.embed(data, cos1) is data
    assert chart.restrict(data, cos1) is data


def test_mode_chart_is_identity_on_periodic_spaces(kit_2d):
    grid, kit, _ = kit_2d
    chart = ModeChart(grid)
    data = jnp.arange(float(N2 * (N2 // 2 + 1))).reshape(
        N2 // 2 + 1, N2)
    # bitwise pass-through: the identical array object comes back
    assert chart.embed(data, kit.coeff("p")) is data
    assert chart.restrict(data, kit.coeff("p")) is data


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


# ================================================================
#  Coefficient-frame override (the distributed eigenmode hook)
# ================================================================
def test_coeff_spaces_override_reads_the_supplied_frame(kit_2d):
    # the frame hook: coeff() (and hence every diff / interp query,
    # which threads on that frame) reads the supplied override instead
    # of the transform's own codomain; forward / backward are untouched
    grid, base, spaces = kit_2d
    alt = base.coeff("p")  # a valid but different coefficient frame
    kit = GridSymbols(grid, spaces, coeff_spaces=dict.fromkeys(spaces, alt))
    assert kit.coeff("u") is alt.bare
    assert kit.coeff("p") is alt.bare
    # the base kit still reads each component's own transform codomain
    assert base.coeff("u") is not alt.bare
    # the symbol queries thread on the override frame (they build)
    assert kit.diff("x", on="u") is not None
    assert kit.interp("y", on="p") is not None
