"""Tests for fridom.model.io.writer (the tensorstore zarr sink).

The round-trip and resume oracles: the written store opens in xarray
with the ``f.xr`` layout (xgcm dims, coords, attrs) plus a CF time
axis and an ``iteration`` coordinate; values are bitwise-equal to the
in-memory fields; mode="w-" fails loudly on an existing store; append
continues the axis; truncate_after + append reproduce a fork-free
axis; coefficient-space/complex outputs raise.
"""
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import xarray as xr
from jax.experimental import multihost_utils

import fridom.spatial.export as export_module
from fridom.model.clock import Clock
from fridom.model.io import writer as writer_module
from fridom.model.io.triggers import every
from fridom.model.io.writer import Writer
from fridom.spatial.decomposition.tensor import TensorDecomposition
from fridom.spatial.export import export_layout, scalar_to_dataarray
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

DT = 0.5


# ================================================================
#  Duck-typed model / carry (per the wave-5 note)
# ================================================================
class FakeCarry:

    """A minimal model_state: state + clock."""

    def __init__(self, state, clock):
        self.state = state
        self.clock = clock


class FakeTable:

    """A minimal field table for the lifecycle-default selection."""

    def __init__(self, prognostic=(), diagnostic=(), auxiliary=()):
        self.prognostic = tuple(prognostic)
        self.diagnostic = tuple(diagnostic)
        self.auxiliary = tuple(auxiliary)
        self.names = self.prognostic + self.diagnostic + self.auxiliary


class FakeFingerprint:
    def __init__(self, digest):
        self.digest = digest


class FakeModel:

    """The smallest thing Writer.bind consumes."""

    def __init__(self, state, clock, *, table=None, digest=None):
        self._carry = FakeCarry(state, clock)
        if table is not None:
            self.field_table = table
        if digest is not None:
            self.fingerprint = FakeFingerprint(digest)

    @property
    def carry(self):
        return self._carry


def clock_at(it, *, start_date=None):
    """Return a clock ticked to iteration ``it`` (time = it * DT)."""
    clock = Clock(start_date=start_date)
    for _ in range(it):
        clock = clock.tick(DT)
    return clock


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def meshes():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")  # periodic
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    return mx, my


@pytest.fixture
def grid(meshes):
    return Grid(meshes)


@pytest.fixture
def state(grid, meshes):
    mx, my = meshes
    u = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + y, name="u",
                          units="m/s")
    p = grid.create_field(init=lambda x, y: x * y, name="p",
                          units="Pa")
    return VectorField({"u": u, "p": p})


@pytest.fixture
def model(state):
    table = FakeTable(prognostic=("u",), diagnostic=("p",))
    return FakeModel(state, clock_at(0), table=table, digest="deadbeef")


def firing(state, it, **kw):
    """Return a carry firing at iteration ``it``."""
    return FakeCarry(state, clock_at(it, **kw))


# ================================================================
#  Round-trip: opens in xarray with the f.xr layout + time axis
# ================================================================
def test_roundtrip_opens_in_xarray(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["u", "p"], trigger=every(steps=1))
    writer.bind(model)
    for it in (0, 1, 2):
        writer.write(firing(state, it))
    writer.close()

    ds = xr.open_zarr(path, consolidated=False)
    # xgcm-style dims straight from the function spaces
    assert ds["u"].dims == ("time", "x_right", "y")
    assert ds["p"].dims == ("time", "x", "y")
    # coordinates: spatial + CF time + iteration (promoted)
    assert "iteration" in ds.coords
    assert "time" in ds.coords
    assert ds.coords["x_right"].attrs["c_grid_axis_shift"] == 0.5
    # per-variable FieldMetadata attrs
    assert ds["u"].attrs["units"] == "m/s"
    assert ds["p"].attrs["units"] == "Pa"
    # provenance
    assert ds.attrs["fridom_fingerprint"] == "deadbeef"
    assert ds.attrs["Conventions"] == "CF-1.10"
    # exact iteration / time axes
    assert ds["iteration"].values.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        ds["time"].values, np.array([0.0, DT, 2 * DT]))
    # bitwise-equal values, every slice
    u_ref = np.asarray(state["u"].data)
    p_ref = np.asarray(state["p"].data)
    for k in range(3):
        np.testing.assert_array_equal(ds["u"].values[k], u_ref)
        np.testing.assert_array_equal(ds["p"].values[k], p_ref)
    # spatial coords equal the grid evaluation nodes (the writer's
    # suffixed-name export; ScalarField.xr itself uses plain names)
    da_u = scalar_to_dataarray(state["u"])
    np.testing.assert_array_equal(
        ds.coords["x_right"].values, da_u.coords["x_right"].values)


def test_calendar_time_axis(tmp_path, model, state):
    path = tmp_path / "cal.zarr"
    start = np.datetime64("2020-01-01T00:00:00")
    cal_model = FakeModel(state, clock_at(0, start_date=start),
                          table=model.field_table)
    writer = Writer(path, fields=["p"], trigger=every(steps=1))
    writer.bind(cal_model)
    writer.write(firing(state, 4, start_date=start))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    # CF decoding turned the time axis into a datetime
    assert ds["time"].dtype.kind == "M"
    assert ds["time"].values[0] == start + np.timedelta64(
        round(4 * DT * 1e9), "ns")
    # whole-second reference date (ns precision breaks CFTime readers)
    raw = xr.open_zarr(path, consolidated=False, decode_times=False)
    assert raw["time"].attrs["units"] == (
        "seconds since 2020-01-01T00:00:00")


# ================================================================
#  Mode vocabulary
# ================================================================
def test_mode_w_minus_fails_on_existing(tmp_path, model):
    path = tmp_path / "out.zarr"
    Writer(path, fields=["p"], trigger=every(steps=1)).bind(model)
    with pytest.raises(FileExistsError):
        Writer(path, fields=["p"], trigger=every(steps=1)).bind(model)


def test_mode_w_clobbers(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    first = Writer(path, fields=["p"], trigger=every(steps=1))
    first.bind(model)
    first.write(firing(state, 0))
    first.close()
    second = Writer(path, fields=["p"], mode="w",
                    trigger=every(steps=1))
    second.bind(model)  # clobbers; fresh axis
    second.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["time"].shape == (0,)


def test_double_bind_raises(tmp_path, model):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1))
    writer.bind(model)
    with pytest.raises(RuntimeError, match="already bound"):
        writer.bind(model)


# ================================================================
#  Append continues the axis (resume without truncation)
# ================================================================
def test_append_continues_axis(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    first = Writer(path, fields=["p"], trigger=every(steps=1))
    first.bind(model)
    for it in (0, 1, 2):
        first.write(firing(state, it))
    first.close()

    resumed = Writer(path, fields=["p"], mode="a",
                     trigger=every(steps=1))
    resumed.bind(model)
    resumed.write(firing(state, 3))
    resumed.close()

    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1, 2, 3]
    assert np.all(np.diff(ds["iteration"].values) > 0)  # fork-free


# ================================================================
#  truncate_after + append reproduce a fork-free axis
# ================================================================
def test_truncate_after_is_forkfree(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1))
    writer.bind(model)
    for it in (0, 1, 2, 3):
        writer.write(firing(state, it))
    # resume from the snapshot at iteration 1: drop 2, 3
    writer.truncate_after(1)
    writer.write(firing(state, 2))
    writer.close()

    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1, 2]
    assert np.all(np.diff(ds["iteration"].values) > 0)


def test_truncate_then_reopen_append(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1))
    writer.bind(model)
    for it in (0, 1, 2, 3):
        writer.write(firing(state, it))
    writer.close()
    # the run-machinery resume shape: reopen append, truncate, write
    resumed = Writer(path, fields=["p"], mode="a",
                     trigger=every(steps=1))
    resumed.bind(model)
    resumed.truncate_after(1)
    resumed.write(firing(state, 2))
    resumed.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1, 2]


# ================================================================
#  Lifecycle-default selection + derived outputs
# ================================================================
def test_lifecycle_default_selection(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    # fields=None -> PROGNOSTIC (u) + DIAGNOSTIC (p); aux opt-in only
    writer = Writer(path, trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 0))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert set(ds.data_vars) == {"u", "p"}


def test_derived_output(tmp_path, model, state):
    path = tmp_path / "out.zarr"
    writer = Writer(
        path, fields=["p"],
        derived={"p2": lambda ms: ms.state["p"] * 2.0},
        trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 0))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert "p2" in ds.data_vars
    np.testing.assert_array_equal(
        ds["p2"].values[0], 2.0 * np.asarray(state["p"].data))


def test_unknown_field_raises(tmp_path, model):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["nope"], trigger=every(steps=1))
    with pytest.raises(ValueError, match="unknown field"):
        writer.bind(model)


def test_default_without_table_raises(tmp_path, state):
    path = tmp_path / "out.zarr"
    bare = FakeModel(state, clock_at(0))  # no field_table
    writer = Writer(path, trigger=every(steps=1))
    with pytest.raises(ValueError, match="explicit fields="):
        writer.bind(bare)


# ================================================================
#  Coefficient-space / complex outputs raise (with a .data pointer)
# ================================================================
def test_complex_field_raises(tmp_path, state):
    path = tmp_path / "out.zarr"
    cstate = VectorField({"c": state["p"].as_complex()})
    model = FakeModel(cstate, clock_at(0))
    writer = Writer(path, fields=["c"], trigger=every(steps=1))
    with pytest.raises(NotImplementedError, match=r"\.data"):
        writer.bind(model)


def test_coefficient_field_raises(tmp_path):
    path = tmp_path / "out.zarr"
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    space = mesh.fourier(origin=mesh.center)
    coeff = grid.create_field(space, init_coeff=lambda kx: kx * 0.0 + 1,
                              name="c")
    model = FakeModel(VectorField({"c": coeff}), clock_at(0))
    writer = Writer(path, fields=["c"], trigger=every(steps=1))
    with pytest.raises(NotImplementedError, match=r"\.data"):
        writer.bind(model)


# ================================================================
#  Walltime triggers rejected at bind (data streams)
# ================================================================
def test_walltime_trigger_rejected(tmp_path, model):
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["p"], trigger=every(walltime="1h"))
    with pytest.raises(ValueError, match="snapshot/action-only"):
        writer.bind(model)


def test_write_before_bind_raises(tmp_path, state):
    writer = Writer(tmp_path / "out.zarr", fields=["p"],
                    trigger=every(steps=1))
    with pytest.raises(RuntimeError, match="not bound"):
        writer.write(firing(state, 0))


# ================================================================
#  The Writer never imports zarr (owner directive)
# ================================================================
def test_writer_module_does_not_import_zarr():
    text = Path(writer_module.__file__).read_text()
    assert "import zarr" not in text


# ================================================================
#  The Writer is xarray-free at runtime (bind = layout, write = sink)
# ================================================================
def test_writer_bind_write_close_without_xarray(
        tmp_path, model, state, monkeypatch):
    # gather_free_output_plan.md phase 2: neither bind nor write may
    # reach export's xarray import hook. Break it, then run the full
    # cycle: if any path called scalar_to_dataarray it would raise.
    def _no_xarray():
        raise ImportError("xarray is banned for this test")

    monkeypatch.setattr(export_module, "_import_xarray", _no_xarray)
    path = tmp_path / "out.zarr"
    writer = Writer(path, fields=["u", "p"], trigger=every(steps=1))
    writer.bind(model)
    for it in (0, 1):
        writer.write(firing(state, it))
    writer.truncate_after(0)
    writer.write(firing(state, 1))
    writer.close()
    # the store is intact and bitwise-correct (read with the real
    # xarray, which the monkeypatch does not touch)
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1]
    u_ref = np.asarray(state["u"].data)
    np.testing.assert_array_equal(ds["u"].values[0], u_ref)


# ================================================================
#  Gather-free write path (phase 3: shard-wise tensorstore writes)
# ================================================================
# These build the field's grid over *every* available device: a single
# shard in the default suite (device-count agnostic) and a genuine 1-D
# sharding under the forced-devices suite
# (XLA_FLAGS=--xla_force_host_platform_device_count=4), where the
# per-shard true-DOF tiling, cell padding, and the empty-tile skip are
# actually exercised. n_cells 15 (periodic x) is non-divisible by 4 and
# still shards under the default halo-2 negotiation (per-shard blocks
# 4, 4, 4, 3 — the last one short), so the file's spatial shape proves
# padding never reaches the store.
class _EmptyRegistry:

    """A halo-free dispatch: negotiate then shards at width 0."""

    def items(self):
        return iter(())


def _staggered_model(device_ids):
    """Build a non-divisible sharded grid with staggered outputs."""
    mx = IntervalMesh(15, (0.0, 1.0), name="x")  # periodic
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((mx, my), device_ids=device_ids)
    u = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + 2.0 * y, name="u",
                          units="m/s")
    p = grid.create_field(init=lambda x, y: x * y, name="p", units="Pa")
    state = VectorField({"u": u, "p": p})
    table = FakeTable(prognostic=("u",), diagnostic=("p",))
    model = FakeModel(state, clock_at(0), table=table, digest="beef")
    return model, state


def test_write_never_gathers(tmp_path, forced_devices, monkeypatch):
    # phase 3: the write path writes each shard's own true-DOF tile and
    # never forms the global array. Break both the export gather helper
    # and the decomposition's gather after bind; the full cycle must
    # still succeed and the store must be bitwise-correct.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    model, state = _staggered_model(None)
    u_ref = np.asarray(state["u"].data)
    p_ref = np.asarray(state["p"].data)
    path = tmp_path / "nogather.zarr"
    writer = Writer(path, fields=["u", "p"], trigger=every(steps=1))
    writer.bind(model)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the write path must not gather")

    monkeypatch.setattr(export_module, "gathered_values", _boom)
    monkeypatch.setattr(TensorDecomposition, "gather", _boom)
    for it in (0, 1):
        writer.write(firing(state, it))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    for k in range(2):
        np.testing.assert_array_equal(ds["u"].values[k], u_ref)
        np.testing.assert_array_equal(ds["p"].values[k], p_ref)


def test_chunk_default_is_write_aligned(tmp_path, forced_devices):
    # default spatial chunks come from the decomposition's write-aligned
    # grid (chunk_hint mapped through kept_axes), asserted through the
    # public API so the same test is meaningful under the forced-4 suite
    # (per-shard block chunks) and the single-device suite (full extent).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    model, state = _staggered_model(None)
    path = tmp_path / "chunks.zarr"
    writer = Writer(path, fields=["u"], trigger=every(steps=1))
    writer.bind(model)
    writer.close()
    zarray = json.loads((path / "u" / ".zarray").read_text())
    field = state["u"]
    layout = export_layout(field)
    hint = field.grid.decomposition.chunk_hint(field.function_space)
    expected = [hint[axis] for axis in layout.kept_axes]
    assert zarray["chunks"] == [1, *expected]
    assert zarray["chunks"][0] == 1  # time chunk stays 1


def test_writer_is_device_count_invariant(tmp_path, forced_devices):
    # the key padding test: the sharded store (all devices) is bitwise
    # identical to the single-device store, the file's spatial shape is
    # the true shape (padding provably absent), and the time/iteration
    # axes agree — mirrors test_export.py's device-count invariance.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_model, many_state = _staggered_model(None)
    one_model, one_state = _staggered_model((0,))
    stores = {}
    for tag, model, state in (("many", many_model, many_state),
                              ("one", one_model, one_state)):
        path = tmp_path / f"{tag}.zarr"
        writer = Writer(path, fields=["u", "p"], trigger=every(steps=1))
        writer.bind(model)
        for it in (0, 1):
            writer.write(firing(state, it))
        writer.close()
        stores[tag] = xr.open_zarr(path, consolidated=False)
    dm, do = stores["many"], stores["one"]
    for name in ("u", "p"):
        true_shape = np.asarray(many_state[name].data).shape
        assert dm[name].shape == do[name].shape
        assert dm[name].shape[1:] == true_shape  # no padding in the file
        np.testing.assert_array_equal(dm[name].values, do[name].values)
    np.testing.assert_array_equal(dm["time"].values, do["time"].values)
    assert (dm["iteration"].values.tolist()
            == do["iteration"].values.tolist())


def test_empty_last_shard_tile_is_skipped(tmp_path, forced_devices):
    # a bounded Inner space (n = n_cells - 1) over a halo-free width-0
    # sharding: at n_cells=10 on P=4 the last of the four shards holds
    # zero true DOFs (a legal empty tile), which the sink skips. The
    # written values must still match the gathered reference exactly.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    ids = tuple(range(jax.device_count()))
    mx = IntervalMesh(10, (0.0, 1.0), periodic=False, name="x")
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((mx, my), dispatch=_EmptyRegistry(), device_ids=ids)
    q = grid.create_field(mx.inner * my.center,
                          init=lambda x, y: x + y, name="q")
    state = VectorField({"q": q})
    model = FakeModel(state, clock_at(0))
    ref = np.asarray(q.data)
    path = tmp_path / "empty.zarr"
    writer = Writer(path, fields=["q"], trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 0))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["q"].shape == (1, *ref.shape)
    np.testing.assert_array_equal(ds["q"].values[0], ref)


def test_replicated_derived_output_written_everywhere(
        tmp_path, forced_devices):
    # the end-to-end regression (gather_free_output_plan): a derived
    # output whose evaluator returns a FULLY REPLICATED field. The
    # production repro was state.rel_vort.to(center) coming back on a
    # PartitionSpec() sharding — one replica-0 shard whose window spans
    # every storage block, so the sink must write the whole domain, not
    # just block 0 (3/4 of the file was silently zeros before the fix).
    # On one device this degenerates; under forced-4 it is the repro and
    # must execute (not skip).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    model, state = _staggered_model(None)
    p = state["p"]
    decomp = p.grid.decomposition
    replicated = jax.device_put(
        p.storage,
        jax.sharding.NamedSharding(
            decomp.device_mesh, jax.sharding.PartitionSpec()))
    ref = np.asarray(p.data)  # the original, non-replicated values

    def _replicated(model_state):
        return model_state.state["p"].with_storage(replicated)

    path = tmp_path / "replicated.zarr"
    writer = Writer(path, fields=[], derived={"pr": _replicated},
                    trigger=every(steps=1))
    writer.bind(model)
    for it in (0, 1):
        writer.write(firing(state, it))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["pr"].shape == (2, *ref.shape)
    for k in range(2):
        # equal everywhere — in particular beyond the first shard's
        # window, which the pre-fix single-block write left zero.
        np.testing.assert_array_equal(ds["pr"].values[k], ref)


# ================================================================
#  async_writes: deferred (overlapping) writes, one firing in flight
# ================================================================
def test_async_writes_match_sync(tmp_path, model, state):
    # the deferred writer must produce a byte-identical store
    fields = ["u", "p"]
    firings = (0, 1, 2, 3)

    def run(path, *, async_writes):
        writer = Writer(path, fields=fields, trigger=every(steps=1),
                        async_writes=async_writes)
        writer.bind(model)
        for it in firings:
            writer.write(firing(state, it))
        writer.close()
        return xr.open_zarr(path, consolidated=False)

    sync = run(tmp_path / "sync.zarr", async_writes=False)
    lazy = run(tmp_path / "async.zarr", async_writes=True)
    for name in (*fields, "time", "iteration"):
        np.testing.assert_array_equal(
            lazy[name].values, sync[name].values)


def test_async_defers_writes_until_drained(tmp_path, model, state):
    # the blocking default leaves nothing pending; async holds the
    # last firing until the next one (or close) drains it.
    sync = Writer(tmp_path / "s.zarr", fields=["p"],
                  trigger=every(steps=1))
    sync.bind(model)
    sync.write(firing(state, 0))
    assert sync._pending is None
    sync.close()

    lazy = Writer(tmp_path / "a.zarr", fields=["p"],
                  trigger=every(steps=1), async_writes=True)
    lazy.bind(model)
    lazy.write(firing(state, 0))
    assert lazy._pending is not None
    lazy.close()
    assert lazy._pending is None
    ds = xr.open_zarr(tmp_path / "a.zarr", consolidated=False)
    np.testing.assert_array_equal(
        ds["p"].values[0], np.asarray(state["p"].data))


def test_async_backpressure_keeps_one_firing_in_flight(
        tmp_path, model, state):
    # each firing drains the previous one, so the outstanding-write
    # count never grows across firings (bounded source buffers).
    lazy = Writer(tmp_path / "bp.zarr", fields=["u", "p"],
                  trigger=every(steps=1), async_writes=True)
    lazy.bind(model)
    lazy.write(firing(state, 0))
    after_first = len(lazy._pending[0])  # queued var writes of one firing
    assert after_first  # something is genuinely in flight
    for it in (1, 2, 3):
        lazy.write(firing(state, it))
        # constant: the previous firing was drained at the top of write
        assert len(lazy._pending[0]) == after_first
    lazy.close()


def test_async_truncate_after_is_forkfree(tmp_path, model, state):
    # truncate_after drains before it resizes the arrays down
    path = tmp_path / "trunc.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1),
                    async_writes=True)
    writer.bind(model)
    for it in (0, 1, 2):
        writer.write(firing(state, it))
    writer.truncate_after(1)
    assert writer._pending is None
    writer.write(firing(state, 2))
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1, 2]


def test_async_crash_leaves_uncommitted_tail_and_reopen_drops_it(
        tmp_path, model, state):
    # a hard kill (no close, no final drain) must never leave a real
    # iteration label on the in-flight slice: it keeps the fill
    # sentinel, and reopen trims it so the append is gap-free.
    path = tmp_path / "crash.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1),
                    async_writes=True)
    writer.bind(model)
    writer.write(firing(state, 0))
    writer.write(firing(state, 1))  # firing 1 is now in flight
    del writer  # simulate the process dying before the next drain

    # firing 0 committed its label; firing 1's is the sentinel, so the
    # slice reads as never-written rather than real-label-on-garbage.
    it_store = writer_module._open_array(path / "iteration")
    raw = np.asarray(it_store.read().result())
    assert raw.tolist() == [0, writer_module._UNWRITTEN]

    resumed = Writer(path, fields=["p"], mode="a",
                     trigger=every(steps=1), async_writes=True)
    resumed.bind(model)
    assert resumed._n == 1  # the crashed tail was dropped on reopen
    resumed.write(firing(state, 1))
    resumed.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds["iteration"].values.tolist() == [0, 1]
    np.testing.assert_array_equal(
        ds["p"].values[0], np.asarray(state["p"].data))


def test_committed_length_counts_up_to_the_sentinel():
    assert writer_module._committed_length(np.array([], np.int64)) == 0
    assert writer_module._committed_length(np.array([0, 1, 2])) == 3
    tail = np.array([0, 1, writer_module._UNWRITTEN])
    assert writer_module._committed_length(tail) == 2


def test_block_writes_reports_the_step_on_failure(tmp_path):
    class Boom:
        def result(self):
            raise OSError("disk full")

    with pytest.raises(RuntimeError, match="at step 7"):
        writer_module._block_writes([(Boom(), None)], tmp_path, 7)


# ================================================================
#  Multi-process seams (_barrier is a no-op single-process)
# ================================================================
def test_barrier_is_noop_single_process(monkeypatch):
    # count == 1: no collective is fired (single-process stays plain)
    monkeypatch.setattr(writer_module, "_process_count", lambda: 1)
    writer_module._barrier("noop")  # must not raise / touch jax


def test_barrier_syncs_when_distributed(monkeypatch):
    # count > 1: the jax collective fires with the given tag
    monkeypatch.setattr(writer_module, "_process_count", lambda: 2)
    called = []
    monkeypatch.setattr(multihost_utils, "sync_global_devices",
                        called.append)
    writer_module._barrier("sync-tag")
    assert called == ["sync-tag"]


# ================================================================
#  Multi-process coordination (simulated ranks via the seams)
# ================================================================
# These monkeypatch the process seams and the tensorstore/JSON layer to
# drive one simulated rank of a multi-process world in a single real
# process: rank 0 owns every metadata mutation (create, resize, the
# time/iteration labels); every other rank opens (not creates) the
# skeleton and writes only its own disjoint shard tiles. The real
# collective coordination is exercised by the subprocess test
# (test_writer_multiprocess.py) and the srun hardware check.
class _FakeDone:

    """A resolved tensorstore future."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeSlice:

    """A ``store[index]`` view recording its ``.write``."""

    def __init__(self, array, index):
        self._array = array
        self._index = index

    def write(self, block):
        self._array.writes.append((self._index, block))
        return _FakeDone(None)


class _FakeArray:

    """In-memory stand-in for a tensorstore zarr array handle."""

    def __init__(self, read_value=None):
        self.shape = (0,)
        self.labels = {}   # index -> scalar label write (__setitem__)
        self.writes = []   # (index, block) shard writes (.write)
        self._read_value = (np.array([], np.int64)
                            if read_value is None else read_value)

    def resize(self, *, exclusive_max):
        self.shape = tuple(exclusive_max)
        return _FakeDone(self)

    def read(self):
        return _FakeDone(self._read_value)

    def __getitem__(self, index):
        return _FakeSlice(self, index)

    def __setitem__(self, index, value):
        self.labels[index] = value


def _simulate(monkeypatch, *, index, count):
    """Patch the process seams + tensorstore/JSON layer for one rank."""
    barriers, created, opened = [], [], []
    monkeypatch.setattr(writer_module, "_process_index", lambda: index)
    monkeypatch.setattr(writer_module, "_process_count", lambda: count)
    monkeypatch.setattr(writer_module, "_barrier", barriers.append)

    def fake_create(path, *_args, **_kwargs):
        created.append(Path(path).name)
        return _FakeArray()

    def fake_open(path, *, recheck=False):
        opened.append((Path(path).name, recheck))
        return _FakeArray()

    monkeypatch.setattr(writer_module, "_create_array", fake_create)
    monkeypatch.setattr(writer_module, "_open_array", fake_open)
    monkeypatch.setattr(writer_module, "_write_json",
                        lambda *_a, **_k: None)
    return barriers, created, opened


def test_distributed_rank0_creates_skeleton(tmp_path, model, monkeypatch):
    barriers, created, opened = _simulate(monkeypatch, index=0, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["u", "p"],
                    trigger=every(steps=1))
    writer.bind(model)
    assert writer._rank0
    assert writer._distributed
    assert opened == []  # rank 0 opens nothing; it creates the skeleton
    for name in ("time", "iteration", "u", "p", "x_right", "y", "x"):
        assert name in created
    assert "writer-exists" in barriers  # check-then-mutate ordering
    assert "writer-skeleton" in barriers


def test_distributed_nonrank0_opens_not_creates(
        tmp_path, model, monkeypatch):
    barriers, created, opened = _simulate(monkeypatch, index=3, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["u", "p"],
                    trigger=every(steps=1))
    writer.bind(model)
    assert not writer._rank0
    assert writer._distributed
    assert created == []  # non-rank-0 creates nothing
    assert {name for name, _ in opened} == {"time", "iteration", "u", "p"}
    assert all(recheck for _, recheck in opened)  # recheck handles
    assert "writer-skeleton" in barriers
    # the spatial shapes are recorded locally off the layouts
    assert set(writer._spatial) == {"u", "p"}


def test_distributed_rank0_writes_the_labels(
        tmp_path, model, state, monkeypatch):
    barriers, _, _ = _simulate(monkeypatch, index=0, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["u", "p"],
                    trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 5))
    # rank 0 writes the time then iteration labels for slot nt=0
    assert writer._time.labels == {0: 5 * DT}
    assert writer._iteration.labels == {0: 5}
    assert writer._vars["u"].writes  # rank 0 also wrote its shard tile
    assert "writer-resized" in barriers
    assert "writer-data" in barriers


def test_distributed_nonrank0_skips_the_labels(
        tmp_path, model, state, monkeypatch):
    _simulate(monkeypatch, index=3, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["u", "p"],
                    trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 5))
    # a non-rank-0 rank never touches the time/iteration labels ...
    assert writer._time.labels == {}
    assert writer._iteration.labels == {}
    # ... but it does write its own disjoint shard tile
    assert writer._vars["u"].writes


def test_distributed_async_falls_back_to_blocking(
        tmp_path, model, state, monkeypatch):
    _simulate(monkeypatch, index=0, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["p"],
                    trigger=every(steps=1), async_writes=True)
    writer.bind(model)
    assert writer._async_effective is False  # distributed -> blocking
    writer.write(firing(state, 0))
    assert writer._pending is None  # nothing deferred; committed inline


def test_distributed_close_barriers(tmp_path, model, monkeypatch):
    barriers, _, _ = _simulate(monkeypatch, index=0, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["p"],
                    trigger=every(steps=1))
    writer.bind(model)
    writer.close()
    assert "writer-close" in barriers
    assert not writer._bound


def test_distributed_truncate_rank0_resizes(
        tmp_path, model, monkeypatch):
    barriers, _, _ = _simulate(monkeypatch, index=0, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["p"],
                    trigger=every(steps=1))
    writer.bind(model)
    writer._iteration = _FakeArray(read_value=np.array([0, 1, 2, 3]))
    writer.truncate_after(1)
    assert writer._n == 2  # keep iterations <= 1 -> [0, 1]
    assert writer._iteration.shape == (2,)  # rank 0 resized down
    assert "writer-truncate-read" in barriers
    assert "writer-truncate-trim" in barriers


def test_distributed_truncate_nonrank0_does_not_resize(
        tmp_path, model, monkeypatch):
    _simulate(monkeypatch, index=3, count=4)
    writer = Writer(tmp_path / "d.zarr", fields=["p"],
                    trigger=every(steps=1))
    writer.bind(model)
    writer._iteration = _FakeArray(read_value=np.array([0, 1, 2, 3]))
    writer.truncate_after(1)
    assert writer._n == 2  # the kept length agrees on every rank ...
    assert writer._time.shape == (0,)  # ... but only rank 0 resizes


def test_distributed_reopen_rank0_trims(tmp_path, model, monkeypatch):
    path = tmp_path / "d.zarr"
    (path / "u").mkdir(parents=True)
    (path / "p").mkdir()
    barriers, _, opened = _simulate(monkeypatch, index=0, count=4)
    writer = Writer(path, fields=["u", "p"], mode="a",
                    trigger=every(steps=1))
    writer.bind(model)
    assert writer._n == 0  # empty iteration -> committed length 0
    # rank 0 reopens without recheck (it owns the trim)
    assert opened
    assert all(recheck is False for _, recheck in opened)
    assert "writer-reopen-read" in barriers
    assert "writer-reopen-trim" in barriers


def test_distributed_reopen_nonrank0_rechecks(
        tmp_path, model, monkeypatch):
    path = tmp_path / "d.zarr"
    (path / "u").mkdir(parents=True)
    (path / "p").mkdir()
    _barriers, _, opened = _simulate(monkeypatch, index=3, count=4)
    writer = Writer(path, fields=["u", "p"], mode="a",
                    trigger=every(steps=1))
    writer.bind(model)
    assert writer._n == 0
    # a non-rank-0 rank reopens with recheck (to see rank 0's trim)
    assert opened
    assert all(recheck for _, recheck in opened)


def test_open_array_recheck_flag_opens_and_sees_resize(
        tmp_path, model, state):
    # the real recheck handle (the non-rank-0 read path): it opens an
    # existing array and re-reads metadata, so a resize performed
    # through a second handle becomes visible.
    path = tmp_path / "r.zarr"
    writer = Writer(path, fields=["p"], trigger=every(steps=1))
    writer.bind(model)
    writer.write(firing(state, 0))
    writer.close()
    handle = writer_module._open_array(path / "time", recheck=True)
    assert tuple(handle.shape) == (1,)
    assert float(np.asarray(handle.read().result())[0]) == 0.0
    # grow via a second handle; a fresh recheck handle sees the new len
    writer_module._open_array(path / "time").resize(
        exclusive_max=[3]).result()
    reread = writer_module._open_array(path / "time", recheck=True)
    assert tuple(reread.shape) == (3,)
