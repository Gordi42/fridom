"""Tests for fridom.model.io.timeseries (the CSV scalar sink).

The oracles: bind writes a header from the bound columns and one row
per firing (iteration, time, values); values come from scalar
callables over the state (0-d Field or scalar, coerced to float);
truncate_after rewrites the tail keyed on the iteration column; resume
(reopen-append) reproduces a fork-free axis; non-scalar columns raise.
"""
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.clock import Clock
from fridom.model.io.timeseries import TimeSeries
from fridom.model.io.triggers import every
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

DT = 0.25


# ================================================================
#  Duck-typed model / carry
# ================================================================
class FakeCarry:
    def __init__(self, state, clock):
        self.state = state
        self.clock = clock


class FakeModel:
    def __init__(self, state, clock):
        self._carry = FakeCarry(state, clock)

    @property
    def carry(self):
        return self._carry


def clock_at(it):
    clock = Clock()
    for _ in range(it):
        clock = clock.tick(DT)
    return clock


@pytest.fixture
def grid():
    return Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))


@pytest.fixture
def scalar_field(grid):
    # a 0-d field: integrate the one axis away
    return grid.create_field(init=lambda x: x + 1.0).integrate("x")


@pytest.fixture
def model(scalar_field):
    return FakeModel({"e": scalar_field}, clock_at(0))


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.reader(handle))


# ================================================================
#  Header + rows
# ================================================================
def test_header_and_rows(tmp_path, model, scalar_field):
    path = tmp_path / "series.csv"
    ts = TimeSeries(
        path,
        columns={
            "energy": lambda ms: ms.state["e"],
            "const": lambda _ms: 3.5,
        },
        trigger=every(steps=1))
    ts.bind(model)
    for it in (0, 1, 2):
        ts.write(FakeCarry({"e": scalar_field}, clock_at(it)))
    ts.close()

    rows = read_rows(path)
    assert rows[0] == ["iteration", "time", "energy", "const"]
    assert len(rows) == 4  # header + 3
    energy = float(np.asarray(scalar_field.data).reshape(()))
    for k, it in enumerate((0, 1, 2)):
        assert int(rows[k + 1][0]) == it
        assert float(rows[k + 1][1]) == pytest.approx(it * DT)
        assert float(rows[k + 1][2]) == pytest.approx(energy)
        assert float(rows[k + 1][3]) == pytest.approx(3.5)


def test_jax_scalar_coerced(tmp_path, model):
    path = tmp_path / "series.csv"
    ts = TimeSeries(
        path, columns={"j": lambda _ms: jnp.asarray(2.0)},
        trigger=every(steps=1))
    ts.bind(model)
    ts.write(FakeCarry({}, clock_at(0)))
    ts.close()
    assert float(read_rows(path)[1][2]) == pytest.approx(2.0)


# ================================================================
#  truncate_after: tail rewrite keyed on the iteration column
# ================================================================
def test_truncate_after_tail_rewrite(tmp_path, model, scalar_field):
    path = tmp_path / "series.csv"
    ts = TimeSeries(path, columns={"e": lambda ms: ms.state["e"]},
                    trigger=every(steps=1))
    ts.bind(model)
    for it in (0, 1, 2, 3):
        ts.write(FakeCarry({"e": scalar_field}, clock_at(it)))
    ts.truncate_after(1)
    ts.write(FakeCarry({"e": scalar_field}, clock_at(2)))
    ts.close()

    rows = read_rows(path)
    iters = [int(r[0]) for r in rows[1:]]
    assert iters == [0, 1, 2]
    assert all(np.diff(iters) > 0)  # fork-free


def test_resume_reopen_appends(tmp_path, model, scalar_field):
    path = tmp_path / "series.csv"
    first = TimeSeries(path, columns={"e": lambda ms: ms.state["e"]},
                       trigger=every(steps=1))
    first.bind(model)
    for it in (0, 1, 2, 3):
        first.write(FakeCarry({"e": scalar_field}, clock_at(it)))
    first.close()

    resumed = TimeSeries(path, columns={"e": lambda ms: ms.state["e"]},
                         trigger=every(steps=1))
    resumed.bind(model)  # existing header matches -> append
    resumed.truncate_after(1)
    resumed.write(FakeCarry({"e": scalar_field}, clock_at(2)))
    resumed.close()

    iters = [int(r[0]) for r in read_rows(path)[1:]]
    assert iters == [0, 1, 2]


def test_resume_header_mismatch_raises(tmp_path, model):
    path = tmp_path / "series.csv"
    TimeSeries(path, columns={"a": lambda _ms: 1.0},
               trigger=every(steps=1)).bind(model)
    other = TimeSeries(path, columns={"b": lambda _ms: 1.0},
                       trigger=every(steps=1))
    with pytest.raises(ValueError, match="header"):
        other.bind(model)


# ================================================================
#  Scalar-ness check (hinted error) + protocol guards
# ================================================================
def test_non_scalar_field_raises(tmp_path, grid):
    # a full (non-reduced) field is not a scalar column
    field = grid.create_field(init=lambda x: x)
    model = FakeModel({"f": field}, clock_at(0))
    ts = TimeSeries(path=tmp_path / "s.csv",
                    columns={"f": lambda ms: ms.state["f"]},
                    trigger=every(steps=1))
    with pytest.raises(ValueError, match="not a scalar"):
        ts.bind(model)


def test_empty_columns_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one column"):
        TimeSeries(tmp_path / "s.csv", columns={},
                   trigger=every(steps=1))


def test_walltime_trigger_rejected(tmp_path, model):
    ts = TimeSeries(tmp_path / "s.csv", columns={"a": lambda _ms: 1.0},
                    trigger=every(walltime="1h"))
    with pytest.raises(ValueError, match="snapshot/action-only"):
        ts.bind(model)


def test_write_before_bind_raises(tmp_path):
    ts = TimeSeries(tmp_path / "s.csv", columns={"a": lambda _ms: 1.0},
                    trigger=every(steps=1))
    with pytest.raises(RuntimeError, match="not bound"):
        ts.write(FakeCarry({}, clock_at(0)))
