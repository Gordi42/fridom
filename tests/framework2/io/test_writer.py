"""Tests for fridom.framework2.io.writer (the tensorstore zarr sink).

The round-trip and resume oracles: the written store opens in xarray
with the ``f.xr`` layout (xgcm dims, coords, attrs) plus a CF time
axis and an ``iteration`` coordinate; values are bitwise-equal to the
in-memory fields; mode="w-" fails loudly on an existing store; append
continues the axis; truncate_after + append reproduce a fork-free
axis; coefficient-space/complex outputs raise.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.io import writer as writer_module
from fridom.framework2.io.triggers import every
from fridom.framework2.io.writer import Writer
from fridom.framework2.model.clock import Clock

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
    # spatial coords equal the grid evaluation nodes (via f.xr)
    da_u = state["u"].xr
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
