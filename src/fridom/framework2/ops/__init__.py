"""
The operations namespace (``fr.ops``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/io_ops.md``.
Wave 4 C adds ``WalltimeGuard``/``ProgressReporter``/``ChunkStats``
from ``protocols``; Wave 5 A adds ``Session``. ``PendingAdvance``
stays reserved (CS-5), never built in iteration 1.
"""
from lazypimp import setup

base = "fridom.framework2.ops"

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
