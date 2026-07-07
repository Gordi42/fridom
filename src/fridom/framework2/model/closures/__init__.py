"""
The closure namespace (``fr.closures``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/module.md``.
Wave 6 B adds ``ClosureBase`` from ``base``.
"""
from lazypimp import setup

base = "fridom.framework2.model.closures"

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
