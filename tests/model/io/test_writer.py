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
