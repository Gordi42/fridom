"""
The time-stepper namespace (``fr.time_steppers``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/time_steppers.md``.
Re-exports grow wave by wave: Wave 4 adds ``AdamBashforth``; Wave 5
adds ``ExplicitRungeKutta``, ``LowStorageRK3``, ``tableaus``,
``IMEXMultistep``, ``CNAB2``, ``SBDF2``.
"""
from lazypimp import setup

base = "fridom.framework2.model.time_steppers"

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
