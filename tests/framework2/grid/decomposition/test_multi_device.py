"""
Device-count invariance of the decomposition (ROADMAP task 1.5).

Every unmarked test compares an auto-negotiated grid (all available
devices) against an explicit one-device grid and must therefore pass
on any device count; under the forced-devices suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) the auto grid is genuinely sharded
and the comparisons prove bitwise device-count invariance through
the ``shard_map`` + ``ppermute`` halo exchange. ``multi_device``
marked tests additionally inspect the blocked storage itself.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.spaces.nodal import NodeSet


def build_grid(device_ids):
    mx = IntervalMesh(8, (0.0, 1.0), name="x")  # periodic
    my = IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")
    return Grid((mx, my), device_ids=device_ids)


@pytest.fixture
def grids(forced_devices):
    if forced_devices is not None:
        # the forced suite must actually see the devices (fail, not
        # skip, when the forcing did not take effect)
        assert jax.device_count() == forced_devices
    return build_grid(None), build_grid((0,))


def bitwise(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def init(x, y):
    return jnp.sin(2.0 * jnp.pi * x) + jnp.cos(y) + x * y


# ================================================================
#  Device-count invariance (bitwise-equal gathered results)
# ================================================================
def test_create_field_init_is_device_count_invariant(grids):
    many, one = grids
    f_many = many.create_field(init=init)
    f_one = one.create_field(init=init)
    assert bitwise(f_many.data, f_one.data)
    gathered = many.decomposition.gather(
        f_many._data, f_many.function_space)
    assert bitwise(gathered, f_one.data)


def test_random_normal_is_device_count_invariant(grids):
    many, one = grids
    space_many = many.create_field().function_space
    space_one = one.create_field().function_space
    for seed in (0, 7):
        r_many = many.random.normal(space_many, seed=seed)
        r_one = one.random.normal(space_one, seed=seed)
        assert bitwise(r_many.data, r_one.data)


def test_diff_chains_periodic_and_bounded(grids):
    many, one = grids
    f_many = many.create_field(init=init)
    f_one = one.create_field(init=init)
    for path in (
        lambda f: f.diff("x"),                    # periodic exchange
        lambda f: f.diff("y"),                    # bounded exchange
        lambda f: f.diff("x").diff("x"),          # chained syncs
        lambda f: f.diff("y").diff("y"),
        lambda f: f.diff("x").diff("y"),          # mixed axes
    ):
        assert bitwise(path(f_many).data, path(f_one).data)


def test_staggered_pair_diffs_are_invariant(grids):
    # Outer (n + 1) and Inner (n - 1) spaces shard unevenly: the
    # last shard absorbs the surplus/deficit (stagger padding)
    many, one = grids

    def staggered(grid):
        my = grid.factors[1]
        outer = grid.create_field(
            my.outer, init=lambda y: y**3 - 2.0 * y)
        back = outer.diff("y")           # Outer -> Center
        inner = back.diff("y")           # Center -> Inner
        return back, inner, inner.to(my.center)

    for a, b in zip(staggered(many), staggered(one), strict=True):
        assert bitwise(a.data, b.data)


def test_interpolation_and_products_are_invariant(grids):
    many, one = grids

    def compute(grid):
        mx = grid.factors[0]
        f = grid.create_field(init=init)
        g = grid.create_field(init=lambda x, y: 1.0 + x + y)
        return (f * g, f / g, abs(f), f**2,
                f.to(f.function_space.bare.replace(x=mx.right)))

    for a, b in zip(compute(many), compute(one), strict=True):
        assert bitwise(a.data, b.data)


def test_reshard_round_trip_is_invariant(grids):
    many, one = grids
    f_many = many.create_field(init=init)
    f_one = one.create_field(init=init)
    decomp = many.decomposition
    pencil = decomp.layout_for(("x",))
    moved = f_many.reshard(pencil)
    assert moved.function_space.layout == pencil
    back = moved.reshard(decomp.default_layout)
    assert back.function_space is f_many.function_space
    assert bitwise(back.data, f_one.data)
    # a diff computed in the pencil layout is still invariant
    assert bitwise(moved.diff("x").data, f_one.diff("x").data)


def test_reductions_are_device_count_invariant(grids):
    many, one = grids
    f_many = many.create_field(init=init)
    f_one = one.create_field(init=init)
    assert np.allclose(np.asarray(f_many.data.sum()),
                       np.asarray(f_one.data.sum()))
    assert not bool(f_many.has_nan())


def test_wider_halo_negotiation_and_order_4_stencils(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    def compute(device_ids):
        mx = IntervalMesh(16, (0.0, 1.0), name="x")
        grid = Grid((mx,), device_ids=device_ids)
        grid.negotiate(halo=HaloSpec({"x": 2}))
        f = grid.create_field(
            init=lambda x: jnp.sin(2.0 * jnp.pi * x))
        return FiniteDifference(order=4)["x"](f)

    assert bitwise(compute(None).data, compute((0,)).data)


# ================================================================
#  Blocked storage internals (genuinely sharded runs only)
# ================================================================
@pytest.mark.multi_device
def test_default_grid_is_genuinely_sharded(grids):
    many, _ = grids
    f = many.create_field(init=init)
    sharding = f._data.sharding
    assert len(sharding.device_set) == jax.device_count()
    spec = sharding.spec
    assert spec[0] == "devices"  # x is the sharded factor


@pytest.mark.multi_device
def test_blocked_ghosts_match_the_single_device_fill():
    # multi-device ghost slots must equal slices of the one-shard
    # halo-extended array (periodic wrap, BC fill, exchanged edges)
    devices = jax.device_count()
    spaces = {
        "periodic center": lambda m: m.center,
        "bounded center": lambda m: m.center,
        "bounded outer": lambda m: m.outer,
        "bounded inner": lambda m: m.inner,
        "dirichlet center": lambda m: m.nodal(
            NodeSet.CENTER, bc=BC.DIRICHLET),
        "neumann center": lambda m: m.nodal(
            NodeSet.CENTER, bc=BC.NEUMANN),
    }
    for label, pick in spaces.items():
        periodic = label.startswith("periodic")
        mesh = IntervalMesh(8, (0.0, 1.0), periodic=periodic,
                            name="x")
        grid_many = Grid((mesh,))
        space = pick(mesh)
        values = jnp.arange(1.0, space.shape[0] + 1.0)
        stored = np.asarray(
            grid_many.create_field(space, data=values)._data)

        mesh_one = IntervalMesh(8, (0.0, 1.0), periodic=periodic,
                                name="x")
        grid_one = Grid((mesh_one,), device_ids=(0,))
        extended = np.asarray(
            grid_one.create_field(pick(mesh_one), data=values)._data)

        n = space.shape[0]
        width = grid_many.decomposition.halo["x"]
        cells = 8 // devices
        block = cells + 1 + 2 * width
        bounds = [min(s * cells, n) for s in range(devices)] + [n]
        for s in range(devices):
            lo, hi = bounds[s], bounds[s + 1]
            piece = stored[s * block:(s + 1) * block]
            # true data, left ghosts, right ghosts against the
            # one-shard frame (offset by its leading halo)
            assert np.array_equal(
                piece[width:width + hi - lo],
                extended[width + lo:width + hi]), label
            assert np.array_equal(
                piece[:width], extended[lo:lo + width]), label
            assert np.array_equal(
                piece[width + hi - lo:2 * width + hi - lo],
                extended[width + hi:2 * width + hi]), label


@pytest.mark.multi_device
def test_coefficient_and_constant_factors_stay_replicated():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    decomp = grid.decomposition
    assert dict(decomp.default_layout.device_axes) == {
        "x": "devices"}
    # coefficient factors are device-local in every layout: no
    # blocking (storage = true shape) and a replicated PartitionSpec
    fourier = mx.fourier(origin=mx.center)
    assert decomp.storage_shape(fourier) == fourier.shape
    assert decomp.sharding(fourier).spec == jax.sharding.PartitionSpec(
        None)
    # constant factors are replicated in every layout
    assert decomp.storage_shape(mx.constant) == (1,)
    assert decomp.sharding(mx.constant).spec == (
        jax.sharding.PartitionSpec(None))


@pytest.mark.multi_device
def test_indivisible_blocking_is_rejected_at_use():
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")  # 5 % devices != 0
    decomp = TensorDecomposition(
        meshes=(mesh,), names=("x",), halo=HaloSpec({"x": 1}),
        layouts=(Layout({"x": "devices"}),),
        device_ids=tuple(range(jax.device_count())))
    with pytest.raises(ValueError, match="divide"):
        decomp.storage_shape(mesh.center)
