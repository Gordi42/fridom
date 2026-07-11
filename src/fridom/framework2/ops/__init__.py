"""
The operations namespace (``fr.ops``).

Description
-----------
Owning class spec: ``design/specs/model/classes/io_ops.md``.
Wave 4 C added ``WalltimeGuard``/``ProgressReporter``/``ChunkStats``
from ``protocols``; Wave 5 A adds ``Session``. ``PendingAdvance``
stays reserved (CS-5), never built in iteration 1.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .protocols import ChunkStats, ProgressReporter, WalltimeGuard
    from .session import Session

base = "fridom.framework2.ops"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base}.protocols": [
        "WalltimeGuard", "ProgressReporter", "ChunkStats"],
    f"{base}.session": ["Session"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
