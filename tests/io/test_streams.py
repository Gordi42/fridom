"""Tests for fridom.io.streams (protocol + errors + column helpers)."""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.io.snapshots import Snapshots
from fridom.io.streams import (
    LEADING_COLUMNS,
    IOCollisionError,
    OutputStream,
    SnapshotMismatchError,
    check_columns,
    coerce_scalar,
    dedupe_streams,
    reject_snapshots_config,
    reject_walltime_trigger,
)
from fridom.io.triggers import at, every


# ================================================================
#  A minimal concrete fake stream
# ================================================================
class FakeStream:

    """The smallest OutputStream implementer."""

    def __init__(self, trigger):
        self.trigger = trigger
        self.bound = None
        self.closed = False

    def bind(self, model):
        reject_walltime_trigger(self.trigger, stream="FakeStream")
        self.bound = model

    def write(self, model_state):
        pass

    def truncate_after(self, iteration):
        pass

    def close(self):
        self.closed = True


class NotAStream:

    """Missing write/truncate_after/close."""

    def __init__(self):
        self.trigger = every(steps=1)

    def bind(self, model):
        pass


# ================================================================
#  Protocol conformance (runtime_checkable)
# ================================================================
def test_fake_stream_satisfies_the_protocol():
    stream = FakeStream(every(steps=1))
    assert isinstance(stream, OutputStream)


def test_incomplete_object_fails_the_protocol():
    assert not isinstance(NotAStream(), OutputStream)
    assert not isinstance(object(), OutputStream)


# ================================================================
#  Walltime-trigger rejection at bind (data streams)
# ================================================================
def test_walltime_trigger_rejected_at_bind_with_snapshots_hint():
    stream = FakeStream(every(walltime="7.5h"))
    with pytest.raises(ValueError, match="Snapshots"):
        stream.bind(model="model")
    assert stream.bound is None


def test_walltime_union_component_also_rejected():
    stream = FakeStream(every(steps=10) | every(walltime="1h"))
    with pytest.raises(ValueError, match="snapshot/action-only"):
        stream.bind(model="model")


def test_model_time_triggers_bind_fine():
    stream = FakeStream(every(steps=10) | at([1.0]))
    stream.bind(model="model")
    assert stream.bound == "model"


# ================================================================
#  IOCollisionError — dedupe by resolved path
# ================================================================
def test_two_distinct_streams_one_path_collide(tmp_path):
    a = FakeStream(every(steps=1))
    b = FakeStream(every(steps=2))
    path = tmp_path / "out.zarr"
    with pytest.raises(IOCollisionError, match="two distinct"):
        dedupe_streams([(a, path), (b, path)])


def test_collision_detected_through_path_spelling(tmp_path):
    a = FakeStream(every(steps=1))
    b = FakeStream(every(steps=2))
    direct = tmp_path / "out.zarr"
    dressed = tmp_path / "sub" / ".." / "out.zarr"
    with pytest.raises(IOCollisionError):
        dedupe_streams([(a, direct), (b, dressed)])


def test_same_stream_twice_dedupes_to_one(tmp_path):
    a = FakeStream(every(steps=1))
    path = tmp_path / "out.zarr"
    kept = dedupe_streams([(a, path), (a, path)])
    assert len(kept) == 1
    assert kept[0][0] is a
    assert kept[0][1] == path.resolve()


def test_distinct_paths_pass_in_order(tmp_path):
    a = FakeStream(every(steps=1))
    b = FakeStream(every(steps=2))
    kept = dedupe_streams(
        [(a, tmp_path / "a.zarr"), (b, tmp_path / "b.csv")])
    assert [stream for stream, _ in kept] == [a, b]


# ================================================================
#  Snapshots rejection outside snapshots=
# ================================================================
def test_snapshots_config_rejected_in_outputs(tmp_path):
    config = Snapshots(tmp_path / "snaps", trigger=every(steps=100))
    with pytest.raises(IOCollisionError, match="snapshots="):
        reject_snapshots_config(config)


def test_snapshots_rejection_names_the_slot(tmp_path):
    config = Snapshots(tmp_path / "snaps", trigger=every(steps=100))
    with pytest.raises(IOCollisionError, match="io="):
        reject_snapshots_config(config, slot="io")


def test_plain_streams_pass_the_snapshots_check():
    reject_snapshots_config(FakeStream(every(steps=1)))


# ================================================================
#  Error types
# ================================================================
def test_error_types_are_exceptions():
    assert issubclass(IOCollisionError, Exception)
    assert issubclass(SnapshotMismatchError, Exception)


# ================================================================
#  The scalar-column helpers (shared by TimeSeries / Series)
# ================================================================
def test_leading_columns_are_the_two_coordinates():
    assert LEADING_COLUMNS == ("iteration", "time")


def test_check_columns_copies_and_preserves_order():
    source = {"b": lambda _ms: 1.0, "a": lambda _ms: 2.0}
    copied = check_columns(source, owner="Sink")
    assert list(copied) == ["b", "a"]
    assert copied is not source


def test_check_columns_rejects_empty():
    with pytest.raises(ValueError, match="Sink needs at least one"):
        check_columns({}, owner="Sink")


def test_check_columns_rejects_non_callable():
    with pytest.raises(TypeError, match="Sink column 'a'"):
        check_columns({"a": 3.0}, owner="Sink")


@pytest.mark.parametrize(
    "value",
    [1.5, np.float64(1.5), np.asarray(1.5), jnp.asarray(1.5),
     np.asarray([1.5])],
    ids=["python", "np-scalar", "np-0d", "jax-0d", "np-len1"])
def test_coerce_scalar_accepts_single_elements(value):
    assert coerce_scalar(value, column="a", owner="Sink") == 1.5


def test_coerce_scalar_rejects_a_non_scalar_array():
    with pytest.raises(ValueError, match=r"Sink column 'a'.*not a scalar"):
        coerce_scalar(np.zeros(3), column="a", owner="Sink")


class FakeField:

    """A duck-typed field: ``data`` plus ``shape``."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape


def test_coerce_scalar_unwraps_a_reduced_field():
    assert coerce_scalar(FakeField([2.5]), column="a",
                         owner="Sink") == 2.5


def test_coerce_scalar_rejects_a_full_field_with_a_hint():
    with pytest.raises(ValueError, match="reduce it first"):
        coerce_scalar(FakeField([1.0, 2.0]), column="a", owner="Sink")
