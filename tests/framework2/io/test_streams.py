"""Tests for fridom.model.io.streams (protocol + errors)."""
import pytest

from fridom.model.io.snapshots import Snapshots
from fridom.model.io.streams import (
    IOCollisionError,
    OutputStream,
    SnapshotMismatchError,
    dedupe_streams,
    reject_snapshots_config,
    reject_walltime_trigger,
)
from fridom.model.io.triggers import at, every


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
