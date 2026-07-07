"""
The state-transform namespace (``fr.transforms``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/transforms.md``.
Wave 7 populates this package: the ``StateTransform`` base + algebra,
``StateSignature``/``TransformInfo``, ``Identity``/``Shift``/
``FixedPoint``, and the Tier-2 presets (``Propagator``,
``TimeAverage``, ``OptimalBalance``). Spectral *operator* transforms
stay in the grid layer; this is the ``State -> State`` namespace.
"""
from lazypimp import setup

base = "fridom.framework2.transforms"

all_modules_by_origin = {}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
