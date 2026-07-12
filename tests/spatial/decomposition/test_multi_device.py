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
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.spaces.nodal import NodeSet


def build_grid(device_ids):
    mx = IntervalMesh(16, (0.0, 1.0), name="x")  # periodic
    my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
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
        lambda f: f.diff("x").diff("y"),          # mixed axes
    ):
        assert bitwise(path(f_many).data, path(f_one).data)
    # the chained bounded syncs run Outer -> Center -> Inner: the
    # exterior-free bounded signatures (BC-free Inner -> Center is
    # gated by R1, boundary_plan.md)
    def outer_chain(grid):
        my = grid.factors[1]
        space = grid.create_field().function_space.bare.replace(
            y=my.outer)
        f = grid.create_field(space, init=init)
        return f.diff("y").diff("y")

    assert bitwise(outer_chain(many).data, outer_chain(one).data)


def test_staggered_pair_diffs_are_invariant(grids):
    # Outer (n + 1) and Inner (n - 1) spaces shard unevenly: the
    # last shard absorbs the surplus/deficit (stagger padding).
    # sin(pi y) vanishes at y = 0 and y = 2 and so does its second
    # derivative, so the Dirichlet retag of the Inner result is the
    # declared-structure route back to centers (BC-free
    # Inner -> Center is gated by R1, boundary_plan.md)
    many, one = grids

    def staggered(grid):
        my = grid.factors[1]
        outer = grid.create_field(
            my.outer, init=lambda y: jnp.sin(jnp.pi * y))
        back = outer.diff("y")           # Outer -> Center
        inner = back.diff("y")           # Center -> Inner
        tagged = inner.retag(
            my.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
        return back, inner, tagged.to(my.center)

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


def test_one_sided_rows_hold_on_the_local_axis(grids):
    # boundary="one_sided" (R2) patches static physical-edge
    # indices: legal on the undistributed bounded y-axis of the
    # default layout, and device-count invariant (the patch reads
    # true DOFs only, never exchanged ghosts)
    many, one = grids
    one_sided = FiniteDifference(order=2, boundary="one_sided")

    def d2(grid):
        f = grid.create_field(init=init)
        return one_sided["y"](f.diff("y"))

    assert bitwise(d2(many).data, d2(one).data)


@pytest.mark.multi_device
def test_one_sided_rows_refuse_a_distributed_axis():
    # a bounded first factor is the sharded axis of the default
    # layout: the one-sided patch must refuse it loudly (the
    # layout="local" requirement, boundary_plan.md 2d)
    my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((my, mx))
    assert dict(grid.decomposition.default_layout.device_axes) == {
        "y": "devices"}
    f = grid.create_field(init=lambda y, x: y * (2.0 - y) + x)
    df = f.diff("y")
    one_sided = FiniteDifference(order=2, boundary="one_sided")
    with pytest.raises(NotImplementedError, match="undistributed"):
        one_sided["y"](df)


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
    many, one = grids
    assert many.decomposition.device_count == jax.device_count()
    assert one.decomposition.device_count == 1
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
        "neumann outer": lambda m: m.nodal(
            NodeSet.OUTER, bc=BC.NEUMANN),
    }
    for label, pick in spaces.items():
        periodic = label.startswith("periodic")
        mesh = IntervalMesh(16, (0.0, 1.0), periodic=periodic,
                            name="x")
        grid_many = Grid((mesh,))
        space = pick(mesh)
        values = jnp.arange(1.0, space.shape[0] + 1.0)
        # consumption-side contract (task 1.8): created fields carry
        # unfilled ghosts; sync explicitly to inspect the fills
        stored = np.asarray(grid_many.sync(
            grid_many.create_field(space, data=values))._data)

        mesh_one = IntervalMesh(16, (0.0, 1.0), periodic=periodic,
                                name="x")
        grid_one = Grid((mesh_one,), device_ids=(0,))
        extended = np.asarray(grid_one.sync(
            grid_one.create_field(pick(mesh_one), data=values))._data)

        n = space.shape[0]
        width = grid_many.decomposition.halo["x"]
        cells = 16 // devices
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
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
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


def test_pad_unpad_round_trip_all_block_shapes(forced_devices):
    # uniform (center) and staggered-uneven (outer/inner) spaces on
    # a sharded bounded axis: the shard-local re-blocking and its
    # global fallback must both be device-count invariant
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    def compute(device_ids):
        my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
        grid = Grid((my,), device_ids=device_ids)
        decomp = grid.decomposition
        outs = []
        for space in (my.center, my.outer, my.inner):
            arr = jnp.arange(1.0, space.shape[0] + 1.0)
            storage = decomp.pad(arr, space)
            outs.append(decomp.gather(storage, space))
            outs.append(decomp.unpad(storage, space))
        return outs

    for a, b in zip(compute(None), compute((0,)), strict=True):
        assert bitwise(a, b)


@pytest.mark.multi_device
def test_local_reblock_plan_exists_only_for_uniform_blocks():
    my = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((my,))
    decomp = grid.decomposition
    layout = decomp.default_layout
    width = decomp.halo["y"]
    cells = 16 // jax.device_count()
    plan = decomp._local_reblock(my.center, layout)
    assert plan is not None
    assert plan.pspec == jax.sharding.PartitionSpec("devices")
    # per-shard: block = cells + 1 + 2 * width
    assert plan.pad_widths == ((width, width + 1),)
    assert plan.true_slices == (slice(width, width + cells),)
    # the plan is cached on the interned (space, layout) key
    assert decomp._local_reblock(my.center, layout) is plan
    # staggered spaces (n = cells * shards +- 1) block unevenly:
    # no plan — pad/unpad fall back to the global re-assembly
    assert decomp._local_reblock(my.outer, layout) is None
    assert decomp._local_reblock(my.inner, layout) is None


@pytest.mark.multi_device
def test_warm_eager_reblocking_adds_zero_compiles(compile_counter):
    # pad/unpad apply the plan's cached jit-wrapped shard_map
    # callables: a per-call closure would re-trace on every eager
    # call and break the compile-count contract (a warmed re-run
    # adds zero compiles — test_run.py, test_imex.py)
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    decomp = grid.decomposition
    space = mx.center
    arr = jnp.arange(1.0, 17.0)
    storage = decomp.pad(arr, space)  # warm the traces
    decomp.unpad(storage, space)
    compile_counter.reset()
    out = decomp.unpad(decomp.pad(arr, space), space)
    assert compile_counter.count == 0
    assert bitwise(out, arr)


@pytest.mark.multi_device
def test_uniform_reblocking_compiles_without_collectives():
    # the diagnosed pathology: global per-block slicing made every
    # unpad/pad round trip cost cross-device all-to-alls; the
    # shard-local plan must compile to zero collectives
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((mx,))
    decomp = grid.decomposition
    space = mx.center

    def round_trip(storage):
        return decomp.pad(decomp.unpad(storage, space), space)

    storage = decomp.pad(jnp.arange(1.0, 17.0), space)
    text = jax.jit(round_trip).lower(storage).compile().as_text()
    for collective in ("all-to-all", "collective-permute",
                       "all-gather", "all-reduce"):
        assert collective not in text
    assert bitwise(round_trip(storage), storage)


@pytest.mark.multi_device
def test_heavy_padding_blocking_is_rejected_at_use():
    # 5 cells over 4 shards: cells=ceil(5/4)=2, (shards-1)*cells=6 >= 5,
    # so a trailing shard would be empty -- rejected loudly at use (a
    # mild non-divisible count would pad instead; see the padded-even
    # tests below)
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")
    decomp = TensorDecomposition(
        meshes=(mesh,), names=("x",), halo=HaloSpec({"x": 1}),
        layouts=(Layout({"x": "devices"}),),
        device_ids=tuple(range(jax.device_count())))
    with pytest.raises(ValueError, match="too heavy"):
        decomp.storage_shape(mesh.center)


# ================================================================
#  Padded-even (non-divisible) ghost sharding
# ================================================================
# 257 % 4 == 1: the exercising case -- center 257, outer 258, inner 256
# (inner is the misaligned n_cells-1 space, Option A / plan section 6).
_NON_DIV = 257


def _direct(n_cells, ids, width=1, *, periodic=False):
    # a decomposition that shards `x` over `ids`, bypassing negotiation
    # (the negotiate side of non-divisible support lands separately) so
    # the padded-even blocking is exercised by direct construction
    mesh = IntervalMesh(n_cells, (0.0, 1.0), periodic=periodic, name="x")
    decomp = TensorDecomposition(
        meshes=(mesh,), names=("x",), halo=HaloSpec({"x": width}),
        layouts=(Layout({"x": "devices"}),), device_ids=ids)
    return mesh, decomp


def test_padded_reblock_round_trip_is_device_count_invariant(
        forced_devices):
    # a non-divisible cell count pads to a uniform per-shard block; the
    # true DOFs must be bitwise device-count invariant (1 vs P) for
    # every space family (center/outer/inner)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mesh_p, decomp_p = _direct(_NON_DIV, tuple(range(jax.device_count())))
    mesh_1, decomp_1 = _direct(_NON_DIV, (0,))
    for pick in (lambda m: m.center, lambda m: m.outer, lambda m: m.inner):
        sp_p, sp_1 = pick(mesh_p), pick(mesh_1)
        arr = jnp.arange(1.0, sp_p.shape[0] + 1.0)
        st_p = decomp_p.pad(arr, sp_p)
        st_1 = decomp_1.pad(arr, sp_1)
        assert bitwise(decomp_p.gather(st_p, sp_p),
                       decomp_1.gather(st_1, sp_1))
        assert bitwise(decomp_p.unpad(st_p, sp_p), arr)


@pytest.mark.multi_device
def test_padded_storage_shape_is_uniform_ceil_blocks():
    # ceil(257/P) cells + stagger slot + 2*halo per shard, P blocks
    devices = jax.device_count()
    width = 1
    mesh, decomp = _direct(_NON_DIV, tuple(range(devices)), width)
    cells = -(-_NON_DIV // devices)
    block = cells + 1 + 2 * width
    for pick in (lambda m: m.center, lambda m: m.outer, lambda m: m.inner):
        assert decomp.storage_shape(pick(mesh)) == (devices * block,)


@pytest.mark.multi_device
def test_padded_local_reblock_plan_exists_for_all_spaces():
    mesh, decomp = _direct(_NON_DIV, tuple(range(jax.device_count())))
    layout = decomp.default_layout
    for pick in (lambda m: m.center, lambda m: m.outer, lambda m: m.inner):
        plan = decomp._local_reblock(pick(mesh), layout)
        assert plan is not None  # padded-even uses the fast path, not
        assert plan.pspec == jax.sharding.PartitionSpec("devices")
        # cached on the interned (space, layout) key
        assert decomp._local_reblock(pick(mesh), layout) is plan


@pytest.mark.multi_device
def test_padded_reblocking_compiles_without_collectives():
    # the perf gate: center (state fields) and outer round-trip through
    # the padded-true fast path with zero collectives -- the padded-even
    # analogue of test_uniform_reblocking_compiles_without_collectives
    mesh, decomp = _direct(_NON_DIV, tuple(range(jax.device_count())))
    for pick in (lambda m: m.center, lambda m: m.outer):
        space = pick(mesh)

        def round_trip(storage, space=space):
            return decomp.pad(decomp.unpad(storage, space), space)

        storage = decomp.pad(jnp.arange(1.0, space.shape[0] + 1.0), space)
        text = jax.jit(round_trip).lower(storage).compile().as_text()
        for collective in ("all-to-all", "collective-permute",
                           "all-gather", "all-reduce"):
            assert collective not in text, (space, collective)


@pytest.mark.multi_device
def test_padded_inner_reduction_is_clean_round_trip_permutes():
    # Option A (plan section 6): the misaligned inner (n_cells-1, here
    # 256, at n_cells % P == 1) keeps the fast path. A storage-frame
    # reduction is a clean all-reduce (no gather / no permute); only the
    # synthetic true-shape round trip costs one neighbour
    # collective-permute -- never all-to-all / all-gather.
    mesh, decomp = _direct(_NON_DIV, tuple(range(jax.device_count())))
    space = mesh.inner
    storage = decomp.pad(jnp.arange(1.0, space.shape[0] + 1.0), space)

    reduce_text = jax.jit(lambda s: s.sum()).lower(
        storage).compile().as_text()
    assert "all-gather" not in reduce_text
    assert "all-to-all" not in reduce_text
    assert "collective-permute" not in reduce_text  # storage-frame reduce

    def round_trip(storage):
        return decomp.pad(decomp.unpad(storage, space), space)

    rt_text = jax.jit(round_trip).lower(storage).compile().as_text()
    assert "all-to-all" not in rt_text     # never the pathology
    assert "all-gather" not in rt_text
    # the expected Option A residual: a cheap neighbour shift (one per
    # trim direction), NOT an all-to-all / all-gather
    assert "collective-permute" in rt_text


@pytest.mark.multi_device
def test_warm_padded_reblocking_adds_zero_compiles(compile_counter):
    # the padded-even plan wraps the shard_map callables with a trailing
    # trim built once and jit-cached: a warm re-run must add 0 compiles
    mesh, decomp = _direct(_NON_DIV, tuple(range(jax.device_count())))
    space = mesh.center
    arr = jnp.arange(1.0, space.shape[0] + 1.0)
    storage = decomp.pad(arr, space)  # warm the traces
    decomp.unpad(storage, space)
    compile_counter.reset()
    out = decomp.unpad(decomp.pad(arr, space), space)
    assert compile_counter.count == 0
    assert bitwise(out, arr)


@pytest.mark.multi_device
def test_padded_sync_ghosts_match_single_device():
    # the sharded halo exchange on a padded-even axis must fill the same
    # ghost slots as the single-device wrap fill (periodic center: the
    # clean case; the last shard is short, ceil-block aligned)
    devices = jax.device_count()
    width = 1
    mesh_p, decomp_p = _direct(_NON_DIV, tuple(range(devices)), width,
                               periodic=True)
    mesh_1, decomp_1 = _direct(_NON_DIV, (0,), width, periodic=True)
    space_p, space_1 = mesh_p.center, mesh_1.center
    n = space_p.shape[0]
    vals = jnp.arange(1.0, n + 1.0)
    stored = np.asarray(
        decomp_p.sync(decomp_p.pad(vals, space_p), space_p))
    extended = np.asarray(
        decomp_1.sync(decomp_1.pad(vals, space_1), space_1))
    cells = -(-n // devices)
    block = cells + 1 + 2 * width
    bounds = [min(s * cells, n) for s in range(devices)] + [n]
    for s in range(devices):
        lo, hi = bounds[s], bounds[s + 1]
        piece = stored[s * block:(s + 1) * block]
        t = hi - lo
        # true content behind the leading halo
        assert np.array_equal(piece[width:width + t],
                              extended[width + lo:width + hi])
        # left ghosts (periodic wrap)
        assert np.array_equal(piece[:width], extended[lo:lo + width])
        # right ghosts sit immediately after the t true DOFs
        assert np.array_equal(piece[width + t:2 * width + t],
                              extended[width + hi:2 * width + hi])


# ================================================================
#  Byte-for-byte no-op on divisible extents (plan Stage 0 / 4)
# ================================================================
_HLO_GOLDEN = Path(__file__).parent / "golden"


def _norm_hlo(text):
    # keep the (normalized) module header + computation body; drop the
    # debug metadata: the FileNames/FileLocations/StackFrames blocks
    # (their paths + source lines shift when code is added above an op)
    # and the inline stack_frame_id (its number depends on how many
    # frames the process has registered, i.e. on suite execution order).
    # What remains is the compiled program -- ops, shapes, collectives,
    # op_names -- which the divisible path must preserve exactly.
    lines = text.splitlines()
    head = re.sub(r"HloModule \S+", "HloModule <m>", lines[0])
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith(("%", "ENTRY")))
    body = "\n".join(lines[start:])
    body = re.sub(r"stack_frame_id=\d+", "stack_frame_id=<n>", body)
    return head + "\n" + body


@pytest.mark.multi_device
def test_divisible_reblock_hlo_is_byte_for_byte_unchanged():
    # THE no-op proof: on a divisible grid the reblock HLO must equal the
    # golden captured from the pre-change tip (the padded-even branch is
    # gated on n_cells % P != 0, so divisible runs the identical path).
    # Regenerate deliberately with FRIDOM_REGEN_HLO_GOLDEN=1 and review
    # the diff -- it must change ONLY on a jax/xla toolchain bump, never
    # from this feature.
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    decomp = Grid((mx,)).decomposition
    space = decomp._meshes[0].center
    true = jnp.arange(1.0, 17.0)
    storage = decomp.pad(true, space)
    ops = {
        "pad": (lambda a: decomp.pad(a, space), true),
        "unpad": (lambda s: decomp.unpad(s, space), storage),
        "round_trip": (
            lambda s: decomp.pad(decomp.unpad(s, space), space), storage),
        "zeros": (lambda _: decomp.zeros(space), storage),
    }
    regen = os.environ.get("FRIDOM_REGEN_HLO_GOLDEN")
    for name, (fn, arg) in ops.items():
        got = _norm_hlo(jax.jit(fn).lower(arg).compile().as_text())
        path = _HLO_GOLDEN / f"divisible_{name}.hlo"
        if regen:
            path.write_text(got)
        assert got == path.read_text(), name
