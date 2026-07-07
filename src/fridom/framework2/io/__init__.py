"""
The IO namespace (``fr.io``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/io_ops.md``.
Re-exports grow wave by wave: Wave 4 C adds the trigger factories
(``every``/``at`` — also re-exported at the framework top level),
``lower_trigger``, ``Snapshots`` + the snapshot store functions, and
``resubmit``; Wave 5 C adds ``Writer`` and ``TimeSeries``.
"""
from lazypimp import setup

base = "fridom.framework2.io"

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
