"""
The IO namespace (``fr.io``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/io_ops.md``.
Wave 4 C adds the trigger factories (``every``/``at`` — also
re-exported at the framework top level), ``lower_trigger``, the
stream protocol + IO errors, ``Snapshots`` + the snapshot store, and
``resubmit``; Wave 5 C adds ``Writer`` and ``TimeSeries``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from . import slurm, snapshots, streams, triggers
    from .slurm import resubmit
    from .snapshots import (
        LeafEntry,
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
    from .streams import (
        IOCollisionError,
        OutputStream,
        SnapshotMismatchError,
    )
    from .triggers import at, every, lower_trigger

base = "fridom.framework2.io"

all_modules_by_origin = {
    base: ["triggers", "streams", "snapshots", "slurm"],
}

all_imports_by_origin = {
    f"{base}.triggers": ["every", "at", "lower_trigger"],
    f"{base}.streams": [
        "OutputStream", "IOCollisionError", "SnapshotMismatchError"],
    f"{base}.snapshots": [
        "Snapshots", "SnapshotManifest", "LeafEntry",
        "write_snapshot", "read_manifest", "read_leaves",
        "find_latest", "rotate", "check_fingerprint", "check_dt"],
    f"{base}.slurm": ["resubmit"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
