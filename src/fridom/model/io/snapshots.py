"""
Restart snapshots.

Description
-----------
The ``Snapshots`` run config, ``SnapshotManifest`` (the versioned
machine-readable header + leaf index, readable WITHOUT leaf IO,
CS-10) and the dumb leaf-blob store functions — orbax-shaped, no
orbax, no pickle: a snapshot directory is a ``manifest.json`` plus
one ``.npy`` file per carry leaf, and the atomic commit is
tmp-dir-then-rename (the rename IS the completeness marker).
Everything here is host-side pure Python + numpy. Owning class
spec: ``design/specs/model/classes/io_ops.md``; fingerprint
scope and the no-pickled-models rule:
``design/specs/model/02_rules.md``.
"""
# Wave 4 C: Snapshots, LeafEntry, SnapshotManifest, store functions
from __future__ import annotations

import json
import re
import shutil
import warnings
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from fridom.model.io.streams import SnapshotMismatchError
from fridom.model.io.triggers import Trigger
from fridom.model.params import TIME_STEP

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# ================================================================
#  Store layout constants
# ================================================================
FORMAT_VERSION: Final[int] = 1

_MANIFEST_NAME: Final[str] = "manifest.json"

# a crashed write leaves a "<name>.tmp" directory: never a snapshot,
# ignored by find_latest, garbage-collected by rotate
_TMP_SUFFIX: Final[str] = ".tmp"

_SLUG_PATTERN: Final = re.compile(r"[^A-Za-z0-9_.-]+")
_SLUG_MAX: Final[int] = 80


# ================================================================
#  Manifest dataclasses
# ================================================================
@dataclass(frozen=True)
class LeafEntry:

    """
    Index entry for one stored carry leaf.

    Parameters
    ----------
    key : str
        The carry tree path of the leaf.
    file : str
        The leaf's ``.npy`` file name, relative to the snapshot
        directory.
    dtype : str
        The numpy dtype name (e.g. ``"float64"``).
    shape : tuple of int
        The TRUE (gathered) global shape — no device identity
        anywhere in the record.
    """

    key: str
    file: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SnapshotManifest:

    """
    Versioned snapshot header + leaf index.

    Description
    -----------
    The header is machine-readable WITHOUT leaf IO (CS-10): clock
    time, iteration, dt (via the provided-parameter values), and
    the fingerprint digest AND its source record (so mismatches
    diff, never silently reuse). IO config is deliberately absent —
    changing output cadence across a resubmit is legal. The
    snapshot never embeds run-loop or driver state (CS-11).

    Parameters
    ----------
    version : int
        The manifest format version (``FORMAT_VERSION``).
    model_name : str or None
        The model's name (log attribution), if any.
    time : float
        The float64 model clock at the snapshot.
    iteration : int
        The iteration count at the snapshot.
    provided_parameters : mapping of str to float
        Recorded provided-parameter values; carries at least
        ``fr.params.TIME_STEP`` ("stepper.dt").
    fingerprint : str
        The restart-fingerprint digest (structure only, never
        leaves).
    fingerprint_record : mapping of str to object
        The fingerprint's diffable source record; values must be
        JSON-serializable (tokens are strings).
    leaves : tuple of LeafEntry
        The leaf index; ``write_snapshot`` rebuilds it from the
        arrays it stores.
    """

    version: int
    model_name: str | None
    time: float
    iteration: int
    provided_parameters: Mapping[str, float]
    fingerprint: str
    fingerprint_record: Mapping[str, object]
    leaves: tuple[LeafEntry, ...] = ()

    @property
    def dt(self) -> float:
        """Shorthand for ``provided_parameters["stepper.dt"]``."""
        try:
            return float(self.provided_parameters[TIME_STEP])
        except KeyError as exc:
            raise KeyError(
                "the manifest records no 'stepper.dt' — "
                "provided_parameters must carry at least "
                "fr.params.TIME_STEP") from exc

    def fingerprint_diff(
        self, record: Mapping[str, object],
    ) -> str | None:
        """Return the human-readable structural diff, or None.

        Description
        -----------
        One line per differing source-record entry ("stepper
        statics differ: cnab2 -> sbdf2"); ``None`` when the
        records match.

        Parameters
        ----------
        record : mapping of str to object
            A model's fingerprint source record (duck-typed plain
            data).

        Returns
        -------
        str or None
            The diff text, or ``None`` on a match.
        """
        stored = dict(self.fingerprint_record)
        fresh = dict(record)
        lines: list[str] = []
        for key in sorted(set(stored) | set(fresh)):
            if key not in fresh:
                lines.append(
                    f"{key} only in the snapshot: {stored[key]}")
            elif key not in stored:
                lines.append(
                    f"{key} only in the model: {fresh[key]}")
            elif stored[key] != fresh[key]:
                lines.append(
                    f"{key} differ: {stored[key]} -> {fresh[key]}")
        return "\n".join(lines) if lines else None


# ================================================================
#  Manifest (de)serialization — JSON, machine-readable
# ================================================================
def _to_json(manifest: SnapshotManifest) -> str:
    """Serialize the manifest to its JSON text."""
    data = {
        "version": int(manifest.version),
        "model_name": manifest.model_name,
        "time": float(manifest.time),
        "iteration": int(manifest.iteration),
        "provided_parameters": {
            str(key): float(value)
            for key, value in manifest.provided_parameters.items()},
        "fingerprint": manifest.fingerprint,
        "fingerprint_record": dict(manifest.fingerprint_record),
        "leaves": [
            {"key": entry.key, "file": entry.file,
             "dtype": entry.dtype, "shape": list(entry.shape)}
            for entry in manifest.leaves],
    }
    return json.dumps(data, indent=2)


def _from_json(text: str, *, source: Path) -> SnapshotManifest:
    """Parse manifest JSON text; check the format version."""
    data = json.loads(text)
    version = int(data["version"])
    if version > FORMAT_VERSION:
        raise ValueError(
            f"snapshot manifest {source} has format version "
            f"{version}; this build reads versions <= "
            f"{FORMAT_VERSION}")
    leaves = tuple(
        LeafEntry(
            key=entry["key"], file=entry["file"],
            dtype=entry["dtype"],
            shape=tuple(int(n) for n in entry["shape"]))
        for entry in data["leaves"])
    return SnapshotManifest(
        version=version,
        model_name=data["model_name"],
        time=float(data["time"]),
        iteration=int(data["iteration"]),
        provided_parameters={
            key: float(value)
            for key, value in data["provided_parameters"].items()},
        fingerprint=data["fingerprint"],
        fingerprint_record=data["fingerprint_record"],
        leaves=leaves)


def _slug(key: str) -> str:
    """Sanitize a leaf key into a file-name fragment."""
    return _SLUG_PATTERN.sub("_", key)[:_SLUG_MAX]


# ================================================================
#  The store functions
# ================================================================
def write_snapshot(
    path: str | Path,
    manifest: SnapshotManifest,
    leaves: Mapping[str, Any],
) -> Path:
    """Atomically commit one snapshot directory.

    Description
    -----------
    Writes the manifest and one ``.npy`` file per leaf into a
    ``<name>.tmp`` sibling directory, then renames — the rename IS
    the completeness marker (``find_latest`` only ever sees
    committed snapshots). The manifest's leaf index is rebuilt from
    ``leaves`` (file names, dtypes, true shapes). Leaves are saved
    with ``allow_pickle=False``: object-dtype leaves are refused
    (the no-pickled-models rule) and the round-trip is bitwise.

    Parameters
    ----------
    path : str or Path
        The target snapshot directory; must not exist yet.
    manifest : SnapshotManifest
        The header to store; its ``leaves`` index is replaced.
    leaves : mapping of str to array-like
        The gathered, true-shape carry leaves keyed by carry tree
        path.

    Returns
    -------
    Path
        The committed snapshot directory.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"snapshot directory {target} already exists; snapshot "
            "paths are unique per firing step")
    arrays: dict[str, np.ndarray] = {}
    for key, leaf in leaves.items():
        arr = np.asarray(leaf)
        if arr.dtype == object:
            raise TypeError(
                f"carry leaf {key!r} has dtype object — snapshots "
                "never pickle (the no-pickled-models rule); leaves "
                "must be plain numeric arrays")
        arrays[key] = arr
    entries = tuple(
        LeafEntry(
            key=key, file=f"leaf_{index:05d}_{_slug(key)}.npy",
            dtype=str(arr.dtype), shape=tuple(arr.shape))
        for index, (key, arr) in enumerate(arrays.items()))
    manifest = replace(manifest, leaves=entries)

    tmp = target.parent / (target.name + _TMP_SUFFIX)
    if tmp.exists():
        shutil.rmtree(tmp)  # stale leftover of a crashed write
    tmp.mkdir(parents=True)
    for entry in entries:
        np.save(tmp / entry.file, arrays[entry.key],
                allow_pickle=False)
    (tmp / _MANIFEST_NAME).write_text(_to_json(manifest))
    tmp.rename(target)  # the atomic commit
    return target


def read_manifest(path: str | Path) -> SnapshotManifest:
    """Read the header + leaf index only — never touches leaf files.

    Description
    -----------
    CS-10: the resume fingerprint check and the coupled cross-model
    consistency assertion read exactly this, with no leaf IO.

    Parameters
    ----------
    path : str or Path
        A committed snapshot directory.

    Returns
    -------
    SnapshotManifest
        The parsed manifest.
    """
    directory = Path(path)
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found — {directory} is not a "
            "committed snapshot directory (tmp dirs and crashed "
            "writes never reach their final name)")
    return _from_json(manifest_path.read_text(), source=manifest_path)


def read_leaves(
    path: str | Path, manifest: SnapshotManifest,
) -> dict[str, np.ndarray]:
    """Load the true-shape leaf arrays named by the manifest.

    Parameters
    ----------
    path : str or Path
        The snapshot directory.
    manifest : SnapshotManifest
        The manifest whose leaf index names the files.

    Returns
    -------
    dict of str to np.ndarray
        The leaves keyed by carry tree path; the round-trip is
        bitwise.
    """
    directory = Path(path)
    out: dict[str, np.ndarray] = {}
    for entry in manifest.leaves:
        arr = np.load(directory / entry.file, allow_pickle=False)
        if (str(arr.dtype) != entry.dtype
                or tuple(arr.shape) != entry.shape):
            raise ValueError(
                f"leaf {entry.key!r} in {directory} does not match "
                f"its manifest entry (found dtype {arr.dtype}, "
                f"shape {tuple(arr.shape)}; the manifest says "
                f"{entry.dtype}, {entry.shape}) — corrupt snapshot")
        out[entry.key] = arr
    return out


def _complete_snapshots(directory: Path) -> list[Path]:
    """List committed snapshot dirs, oldest first.

    A directory counts as a committed snapshot iff its name does
    not end in the tmp suffix and it holds a manifest; ordering is
    by (iteration, mtime).
    """
    if not directory.is_dir():
        return []
    committed = [
        child for child in directory.iterdir()
        if child.is_dir()
        and not child.name.endswith(_TMP_SUFFIX)
        and (child / _MANIFEST_NAME).is_file()]
    return sorted(
        committed,
        key=lambda p: (read_manifest(p).iteration, p.stat().st_mtime))


def find_latest(directory: str | Path) -> Path | None:
    """Return the newest COMPLETE snapshot in a Snapshots directory.

    Description
    -----------
    Only committed snapshots are visible (the atomic rename is the
    commit marker); in-progress/crashed tmp dirs never count.
    "Newest" is the highest manifest iteration — monotone in both
    time directions — with mtime as the tie-breaker.

    Parameters
    ----------
    directory : str or Path
        The Snapshots directory; may not exist yet.

    Returns
    -------
    Path or None
        The newest committed snapshot directory, or ``None``.
    """
    committed = _complete_snapshots(Path(directory))
    return committed[-1] if committed else None


def rotate(directory: str | Path, keep: int) -> None:
    """Delete complete snapshots beyond the newest ``keep``.

    Description
    -----------
    Runs after each committed snapshot: keeps the newest ``keep``
    complete snapshots, deletes the rest, and garbage-collects
    crashed-write tmp dirs. In-progress tmp dirs never count as
    snapshots.

    Parameters
    ----------
    directory : str or Path
        The Snapshots directory; may not exist yet.
    keep : int
        How many complete snapshots to keep; at least 1.
    """
    if isinstance(keep, bool) or not isinstance(keep, int):
        raise TypeError(f"keep must be an int; got {keep!r}")
    if keep < 1:
        raise ValueError(f"keep must be >= 1; got {keep}")
    base = Path(directory)
    if not base.is_dir():
        return
    for child in base.iterdir():
        # garbage-collect crashed writes
        if child.is_dir() and child.name.endswith(_TMP_SUFFIX):
            shutil.rmtree(child)
    committed = _complete_snapshots(base)
    for stale in committed[:-keep]:
        shutil.rmtree(stale)


# ================================================================
#  Resume checks (consumed by load_snapshot / the resume path)
# ================================================================
def check_fingerprint(
    manifest: SnapshotManifest,
    *,
    digest: str,
    record: Mapping[str, object],
) -> None:
    """Check a model's fingerprint against the manifest's.

    Description
    -----------
    A mismatch raises ``SnapshotMismatchError`` whose message diffs
    the source records — never silent reuse of incompatible stepper
    history. Fingerprint scope is structure only; IC differences
    are deliberately invisible.

    Parameters
    ----------
    manifest : SnapshotManifest
        The stored header.
    digest : str
        The assembled model's fingerprint digest.
    record : mapping of str to object
        The assembled model's fingerprint source record.
    """
    if digest == manifest.fingerprint:
        return
    diff = manifest.fingerprint_diff(record)
    if diff is None:
        diff = ("(the source records match; the digests differ — "
                "fingerprint version skew?)")
    raise SnapshotMismatchError(
        "restart fingerprint mismatch — the snapshot was written "
        "by a structurally different model:\n" + diff)


def check_dt(manifest: SnapshotManifest, dt: float) -> None:
    """Check the resuming model's dt against the manifest's.

    Description
    -----------
    The successor of ``run_backward``'s sign re-forcing (02_rules):
    a dt SIGN mismatch errors (``SnapshotMismatchError``); a
    magnitude change only warns (``fr.at`` times may land on
    different realized steps).

    Parameters
    ----------
    manifest : SnapshotManifest
        The stored header (its ``dt`` is read).
    dt : float
        The assembled model's (signed, nonzero) time step.
    """
    stored = manifest.dt
    fresh = float(dt)
    if fresh == 0.0:
        raise ValueError("dt must be nonzero")
    if stored * fresh < 0.0:
        raise SnapshotMismatchError(
            f"time-step sign changed across the restart: the "
            f"snapshot was written with dt={stored}, the model is "
            f"assembled with dt={fresh}. A restart never re-forces "
            "the sign; assemble the model with the intended dt.")
    if stored != fresh:
        warnings.warn(
            f"time-step magnitude changed across the restart: "
            f"{stored} -> {fresh}; fr.at() times may land on "
            "different realized steps.",
            stacklevel=2)


# ================================================================
#  The Snapshots run config
# ================================================================
@dataclass(frozen=True)
class Snapshots:

    """
    Run-config for restart snapshots.

    Description
    -----------
    Where, when, how many, whether to resume, and what to do on
    walltime — pure configuration, consumed by ``run(snapshots=)``
    / ``Session(snapshots=)`` ONLY (the ``io=`` slot and the
    ``outputs=`` tuple reject it: one resume path, never two). The
    trigger may mix model-time components (lowered to steps) with a
    walltime component (``fr.every(walltime="7.5h")``), which the
    ``WalltimeGuard`` evaluates predictively at chunk boundaries.

    Parameters
    ----------
    path : str or Path
        The snapshots directory.
    trigger : Trigger
        When to snapshot.
    keep : int, optional
        Rotation: after each committed snapshot, complete snapshots
        beyond the newest ``keep`` are deleted; ``None`` disables
        rotation.
    resume : bool, optional
        Scan ``path`` for the newest complete snapshot at run start
        and resume from it; a fresh start when none exists
        (default True).
    on_walltime : callable, optional
        Invoked once, on the walltime exit path only, after the
        triggered snapshot has committed (``fr.io.resubmit()`` is
        the shipped one). Never called on normal completion or on
        ``PanicError``.
    """

    path: str | Path
    _: KW_ONLY
    trigger: Trigger
    keep: int | None = None
    resume: bool = True
    on_walltime: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        """Normalize the path and validate the config."""
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.trigger, Trigger):
            raise TypeError(
                "trigger= must be a Trigger built by "
                f"fr.every/fr.at; got {self.trigger!r}")
        if self.keep is not None and (
                isinstance(self.keep, bool)
                or not isinstance(self.keep, int)
                or self.keep < 1):
            raise ValueError(
                f"keep= must be an int >= 1 or None; got "
                f"{self.keep!r}")
        if self.on_walltime is not None and not callable(
                self.on_walltime):
            raise TypeError(
                f"on_walltime= must be callable; got "
                f"{self.on_walltime!r}")
