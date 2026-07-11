"""Tests for fridom.model.io.snapshots (store + config)."""
import dataclasses
import json

import numpy as np
import pytest

from fridom.model.io.snapshots import (
    FORMAT_VERSION,
    SnapshotManifest,
    Snapshots,
    check_dt,
    check_fingerprint,
    find_latest,
    read_leaves,
    read_manifest,
    rotate,
    write_snapshot,
)
from fridom.model.io.streams import SnapshotMismatchError
from fridom.model.io.triggers import every

# ================================================================
#  Fixtures / builders
# ================================================================
RECORD = {
    "stepper statics": "cnab2",
    "modules": "(PressureSolver, Advection)",
    "fields": "(u, v, w, b, p)",
}


def make_manifest(iteration=100, dt=0.1, **over):
    base = {
        "version": FORMAT_VERSION,
        "model_name": "nh",
        "time": iteration * dt,
        "iteration": iteration,
        "provided_parameters": {"stepper.dt": dt},
        "fingerprint": "digest-a",
        "fingerprint_record": dict(RECORD),
        "leaves": (),
    }
    base.update(over)
    return SnapshotManifest(**base)


def make_leaves():
    rng = np.random.default_rng(7)
    u = rng.standard_normal((4, 3))
    u[0, 0] = np.nan
    u[1, 2] = -np.nan  # a second NaN payload, different sign bit
    return {
        "fields/u": u,
        "stepper/history": rng.standard_normal(5),
        "stepper/warmup": np.array(3, dtype=np.int64),
        "clock/time": np.float64(12.5),
    }


# ================================================================
#  Bitwise write/read round-trip
# ================================================================
def test_roundtrip_is_bitwise(tmp_path):
    leaves = make_leaves()
    path = write_snapshot(tmp_path / "snap_100", make_manifest(),
                          leaves)
    loaded = read_leaves(path, read_manifest(path))
    assert set(loaded) == set(leaves)
    for key, original in leaves.items():
        stored = loaded[key]
        arr = np.asarray(original)
        assert stored.dtype == arr.dtype
        assert stored.shape == arr.shape
        # bitwise, including the NaN payloads
        assert stored.tobytes() == arr.tobytes()


def test_manifest_header_roundtrip(tmp_path):
    manifest = make_manifest(iteration=250, dt=0.05)
    path = write_snapshot(tmp_path / "snap_250", manifest,
                          make_leaves())
    loaded = read_manifest(path)
    assert loaded.version == FORMAT_VERSION
    assert loaded.model_name == "nh"
    assert loaded.iteration == 250
    assert loaded.time == pytest.approx(250 * 0.05)
    assert loaded.dt == 0.05
    assert loaded.fingerprint == "digest-a"
    assert dict(loaded.fingerprint_record) == RECORD


def test_leaf_index_records_true_shape_and_dtype(tmp_path):
    path = write_snapshot(tmp_path / "snap_1", make_manifest(),
                          make_leaves())
    index = {e.key: e for e in read_manifest(path).leaves}
    assert index["fields/u"].shape == (4, 3)
    assert index["fields/u"].dtype == "float64"
    assert index["stepper/warmup"].shape == ()
    assert index["stepper/warmup"].dtype == "int64"


def test_dt_property_requires_time_step():
    manifest = make_manifest(provided_parameters={})
    with pytest.raises(KeyError, match=r"stepper\.dt"):
        _ = manifest.dt


# ================================================================
#  read_manifest does no leaf IO
# ================================================================
def test_read_manifest_survives_missing_leaf_file(tmp_path):
    path = write_snapshot(tmp_path / "snap_1", make_manifest(),
                          make_leaves())
    victim = next(path.glob("leaf_*fields_u*.npy"))
    victim.unlink()
    manifest = read_manifest(path)  # header-only: still fine
    assert manifest.iteration == 100
    with pytest.raises(FileNotFoundError):
        read_leaves(path, manifest)


def test_read_leaves_detects_corruption(tmp_path):
    path = write_snapshot(tmp_path / "snap_1", make_manifest(),
                          make_leaves())
    manifest = read_manifest(path)
    victim = next(path.glob("leaf_*fields_u*.npy"))
    np.save(victim, np.zeros(2), allow_pickle=False)
    with pytest.raises(ValueError, match="corrupt"):
        read_leaves(path, manifest)


def test_read_manifest_rejects_future_format_version(tmp_path):
    path = write_snapshot(tmp_path / "snap_1", make_manifest(),
                          make_leaves())
    manifest_path = path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["version"] = FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format version"):
        read_manifest(path)


# ================================================================
#  No pickle, ever
# ================================================================
def test_object_dtype_leaves_are_refused(tmp_path):
    leaves = {"bad": np.array([{"a": 1}], dtype=object)}
    with pytest.raises(TypeError, match="pickle"):
        write_snapshot(tmp_path / "snap_1", make_manifest(), leaves)
    assert find_latest(tmp_path) is None


# ================================================================
#  Atomicity: tmp dir + rename
# ================================================================
def test_write_refuses_existing_target(tmp_path):
    write_snapshot(tmp_path / "snap_1", make_manifest(),
                   make_leaves())
    with pytest.raises(FileExistsError):
        write_snapshot(tmp_path / "snap_1", make_manifest(),
                       make_leaves())


def test_crashed_write_is_invisible_to_find_latest(tmp_path):
    # simulate a crash mid-write: the tmp dir is left behind,
    # manifest included — it must never count as a snapshot
    crashed = tmp_path / "snap_50.tmp"
    crashed.mkdir()
    (crashed / "manifest.json").write_text("{}")
    assert find_latest(tmp_path) is None
    committed = write_snapshot(tmp_path / "snap_10",
                               make_manifest(iteration=10),
                               make_leaves())
    assert find_latest(tmp_path) == committed


def test_rotate_garbage_collects_crashed_tmp_dirs(tmp_path):
    crashed = tmp_path / "snap_50.tmp"
    crashed.mkdir()
    (crashed / "manifest.json").write_text("{}")
    write_snapshot(tmp_path / "snap_10",
                   make_manifest(iteration=10), make_leaves())
    rotate(tmp_path, keep=1)
    assert not crashed.exists()
    assert (tmp_path / "snap_10").is_dir()


def test_stale_tmp_dir_does_not_block_a_new_write(tmp_path):
    stale = tmp_path / "snap_10.tmp"
    stale.mkdir()
    (stale / "junk").write_text("partial")
    path = write_snapshot(tmp_path / "snap_10",
                          make_manifest(iteration=10), make_leaves())
    assert path.is_dir()
    assert not stale.exists()


# ================================================================
#  find_latest / rotate
# ================================================================
def test_find_latest_picks_highest_iteration(tmp_path):
    write_snapshot(tmp_path / "b", make_manifest(iteration=100),
                   make_leaves())
    write_snapshot(tmp_path / "a", make_manifest(iteration=50),
                   make_leaves())
    latest = find_latest(tmp_path)
    assert latest == tmp_path / "b"


def test_find_latest_empty_and_missing_directories(tmp_path):
    assert find_latest(tmp_path) is None
    assert find_latest(tmp_path / "nowhere") is None


def test_rotate_keeps_the_newest_n(tmp_path):
    for iteration in (10, 20, 30):
        write_snapshot(tmp_path / f"snap_{iteration}",
                       make_manifest(iteration=iteration),
                       make_leaves())
    rotate(tmp_path, keep=2)
    assert not (tmp_path / "snap_10").exists()
    assert (tmp_path / "snap_20").is_dir()
    assert (tmp_path / "snap_30").is_dir()
    # surviving snapshots stay loadable
    latest = find_latest(tmp_path)
    assert read_manifest(latest).iteration == 30


def test_rotate_validates_keep(tmp_path):
    with pytest.raises(ValueError, match=">= 1"):
        rotate(tmp_path, keep=0)
    with pytest.raises(TypeError, match="int"):
        rotate(tmp_path, keep=2.0)


def test_rotate_on_missing_directory_is_a_noop(tmp_path):
    rotate(tmp_path / "nowhere", keep=3)


# ================================================================
#  Fingerprint diff + check
# ================================================================
def test_fingerprint_diff_none_on_match():
    assert make_manifest().fingerprint_diff(dict(RECORD)) is None


def test_fingerprint_diff_text_per_entry():
    changed = dict(RECORD, **{"stepper statics": "sbdf2"})
    diff = make_manifest().fingerprint_diff(changed)
    assert diff == "stepper statics differ: cnab2 -> sbdf2"


def test_fingerprint_diff_reports_one_sided_keys():
    changed = dict(RECORD)
    del changed["modules"]
    changed["variant filter"] = "hydrostatic"
    diff = make_manifest().fingerprint_diff(changed)
    assert "modules only in the snapshot" in diff
    assert "variant filter only in the model" in diff


def test_check_fingerprint_passes_on_matching_digest():
    check_fingerprint(make_manifest(), digest="digest-a",
                      record=dict(RECORD))


def test_check_fingerprint_error_prints_the_diff():
    changed = dict(RECORD, **{"stepper statics": "sbdf2"})
    with pytest.raises(SnapshotMismatchError) as err:
        check_fingerprint(make_manifest(), digest="digest-b",
                          record=changed)
    assert "stepper statics differ: cnab2 -> sbdf2" in str(err.value)


# ================================================================
#  dt checks: sign errors, magnitude warns
# ================================================================
def test_check_dt_same_value_is_silent():
    # filterwarnings = error in pyproject: a stray warning fails
    check_dt(make_manifest(dt=0.1), 0.1)


def test_check_dt_sign_mismatch_errors():
    with pytest.raises(SnapshotMismatchError, match="sign"):
        check_dt(make_manifest(dt=0.1), -0.1)
    with pytest.raises(SnapshotMismatchError, match="sign"):
        check_dt(make_manifest(dt=-0.1), 0.1)


def test_check_dt_magnitude_mismatch_warns():
    with pytest.warns(UserWarning, match="magnitude"):
        check_dt(make_manifest(dt=0.1), 0.2)
    with pytest.warns(UserWarning, match="magnitude"):
        check_dt(make_manifest(dt=-0.1), -0.05)


def test_check_dt_rejects_zero():
    with pytest.raises(ValueError, match="nonzero"):
        check_dt(make_manifest(dt=0.1), 0.0)


# ================================================================
#  The Snapshots run config
# ================================================================
def test_snapshots_config_normalizes_and_freezes(tmp_path):
    config = Snapshots(str(tmp_path / "snaps"),
                       trigger=every(steps=100), keep=3)
    assert config.path == tmp_path / "snaps"
    assert config.resume is True
    assert config.on_walltime is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.keep = 5


def test_snapshots_config_accepts_walltime_triggers(tmp_path):
    trigger = every(hours=6) | every(walltime="7.5h")
    config = Snapshots(tmp_path, trigger=trigger,
                       on_walltime=lambda: None)
    assert config.trigger.has_walltime is True


def test_snapshots_config_validation(tmp_path):
    with pytest.raises(TypeError, match="Trigger"):
        Snapshots(tmp_path, trigger="every 100")
    with pytest.raises(ValueError, match="keep"):
        Snapshots(tmp_path, trigger=every(steps=1), keep=0)
    with pytest.raises(TypeError, match="callable"):
        Snapshots(tmp_path, trigger=every(steps=1),
                  on_walltime="resubmit")
