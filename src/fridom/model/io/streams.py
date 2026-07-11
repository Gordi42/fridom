"""
The output-stream protocol.

Description
-----------
``OutputStream`` — the one interface D4 owns; ``Writer`` and
``TimeSeries`` (2.6) and the designed-for capture streams implement
it. Also home to the cluster's error types (``IOCollisionError``,
``SnapshotMismatchError``, registered in the model.md error-type
registry) and the small bind-time helpers the run machinery calls:
walltime-trigger rejection on data streams, ``Snapshots`` rejection
outside ``snapshots=``, and dedupe-by-resolved-path. Owning class
spec: ``design/specs/model/classes/io_ops.md``.
"""
# Wave 4 C: OutputStream, IOCollisionError, SnapshotMismatchError
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fridom.model.io.triggers import Trigger


# ================================================================
#  The protocol
# ================================================================
@runtime_checkable
class OutputStream(Protocol):

    """
    A trigger-driven output stream bound to one model.

    Description
    -----------
    The binding split (amended V-C5): diagnostic-*name* resolution
    happens at assembly (constructor ``io=``) or immediately
    (``run(outputs=...)``); ``bind()`` — store creation, dry
    evaluation — happens at RUN START, never at assembly. Streams
    dedupe by resolved path; two distinct streams targeting one
    path raise ``IOCollisionError`` at bind. Walltime-bearing
    triggers are rejected at bind on data streams (use
    ``fr.io.Snapshots``). ``truncate_after`` is iteration-keyed:
    the iteration coordinate is monotone in both time directions.
    """

    trigger: Trigger

    def bind(self, model: Any) -> None:
        """Resolve names, dry-evaluate, create the store eagerly.

        Called at RUN START, never at assembly.
        """
        ...

    def write(self, model_state: Any) -> None:
        """Write from the boundary-synced carry at a firing boundary.

        Streams fire in deterministic (binding) order.
        """
        ...

    def truncate_after(self, iteration: int) -> None:
        """Drop output past the given iteration coordinate.

        Restart-resume alignment; iteration-keyed per V-S1.
        """
        ...

    def close(self) -> None:
        """Run end or abort; partial output must survive."""
        ...


# ================================================================
#  Error types owned by this cluster
# ================================================================
class IOCollisionError(Exception):

    """
    Two distinct output streams resolved to the same path.

    Description
    -----------
    Raised at bind: streams dedupe by resolved path, and a second
    *distinct* stream targeting one path is an error (06 section
    8.4). Also raised when a ``Snapshots`` config is handed to an
    ``io=``/``outputs=`` slot — snapshots are run-config only, one
    resume path, never two.
    """


class SnapshotMismatchError(Exception):

    """
    Snapshot fingerprint mismatch on resume/load.

    Description
    -----------
    The message DIFFS the fingerprint source records, one line per
    differing entry ("stepper statics differ: cnab2 -> sbdf2") —
    mismatches diff, never silently reuse incompatible stepper
    history. Also raised on a dt SIGN mismatch between the manifest
    and the assembled model (a magnitude change only warns).
    """


# ================================================================
#  Bind-time helpers
# ================================================================
def reject_walltime_trigger(
    trigger: Trigger, *, stream: str | None = None,
) -> None:
    """Reject walltime-bearing triggers on data streams at bind.

    Description
    -----------
    Walltime triggers produce nondeterministic output grids, so
    they are snapshot/action-only; data streams call this in
    ``bind()``.

    Parameters
    ----------
    trigger : Trigger
        The stream's firing trigger.
    stream : str, optional
        A stream name for the error message.
    """
    if not trigger.has_walltime:
        return
    owner = f" on {stream}" if stream else ""
    raise ValueError(
        f"walltime-bearing trigger{owner}: data streams need a "
        "deterministic output grid, so walltime triggers are "
        "snapshot/action-only — move it to "
        "fr.io.Snapshots(trigger=fr.every(walltime=...)).")


def reject_snapshots_config(
    candidate: object, *, slot: str = "outputs",
) -> None:
    """Reject a ``Snapshots`` config outside ``snapshots=``.

    Description
    -----------
    ``Snapshots`` is accepted by ``run(snapshots=)`` /
    ``Session(snapshots=)`` only; the constructor ``io=`` slot and
    the ``outputs=`` tuple reject it — one resume path, never two.

    Parameters
    ----------
    candidate : object
        The object handed to the slot.
    slot : str, optional
        The slot name for the error message (default "outputs").
    """
    from fridom.model.io.snapshots import (  # noqa: PLC0415 — snapshots.py imports this module (SnapshotMismatchError)
        Snapshots,
    )
    if isinstance(candidate, Snapshots):
        raise IOCollisionError(
            f"a Snapshots config was passed via {slot}=; snapshots "
            "are run-config only — pass it as run(snapshots=...) / "
            "Session(snapshots=...). One resume path, never two.")


def dedupe_streams(
    streams: Sequence[tuple[OutputStream, str | Path]],
) -> tuple[tuple[OutputStream, Path], ...]:
    """Dedupe ``(stream, path)`` pairs by resolved path.

    Description
    -----------
    The 06 section 8.4 collision rule: the same stream object
    listed twice collapses to one entry; two *distinct* streams
    resolving to one path raise ``IOCollisionError``.

    Parameters
    ----------
    streams : sequence of (OutputStream, path) pairs
        The union of standing and per-run streams with their
        target paths, in binding order.

    Returns
    -------
    tuple of (OutputStream, Path) pairs
        The deduplicated streams with resolved paths, order
        preserved.
    """
    kept: list[tuple[OutputStream, Path]] = []
    owners: dict[Path, OutputStream] = {}
    for stream, path in streams:
        resolved = Path(path).expanduser().resolve()
        owner = owners.get(resolved)
        if owner is stream:
            continue
        if owner is not None:
            raise IOCollisionError(
                f"two distinct output streams resolved to "
                f"{resolved} — every stream owns its store "
                "exclusively; give each stream its own path.")
        owners[resolved] = stream
        kept.append((stream, resolved))
    return tuple(kept)
