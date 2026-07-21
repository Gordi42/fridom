"""
The shallow-water model.

Description
-----------
A top-level parallel package (renamed onto ``fridom.shallowwater`` at
cutover) that consumes ``fridom`` and provides the
shallow-water vocabulary and physics:

- :class:`~fridom.shallowwater2.state.State` — the ``u``/``v``/``p``
  vocabulary class;
- :mod:`~fridom.shallowwater2.modules` — the core (``sw.Core``,
  also exported at the package root), Coriolis, and Sadourny
  advection modules;
- :func:`~fridom.shallowwater2.model.Model` — the thin preset factory
  (``sw.Model(grid=..., coriolis=..., ...)``);
- :mod:`~fridom.shallowwater2.eigenmodes` — ``from_model``, the
  linear dispersion eigenmodes, and ``eigenbasis`` (the labeled
  channel eigenbasis surface);
- :mod:`~fridom.shallowwater2.channel_eigenmodes` — the labeled
  numeric eigenmodes of the walled channel;
- :mod:`~fridom.shallowwater2.params` — the package parameter names.

Lazy re-exports (mirroring ``fridom.spatial.__init__``): the
framework surface itself stays under ``fridom`` (imported
as ``fr``).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        channel_eigenmodes,
        diagnostics,
        eigenmodes,
        initial_conditions,
        modules,
        params,
        transforms,
        units,
    )

    # import all classes
    from .channel_eigenmodes import ChannelEigenmodes
    from .eigenmodes import eigenbasis
    from .initial_conditions import (
        coherent_eddy,
        jet,
        random_state,
        random_vortical,
        random_waves,
        single_wave,
    )
    from .model import Model
    from .modules.core import Core
    from .state import State

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.shallowwater2"

all_modules_by_origin = {
    base: ["modules", "eigenmodes", "channel_eigenmodes",
           "diagnostics", "params", "transforms",
           "initial_conditions", "units"],
}

all_imports_by_origin = {
    f"{base}.channel_eigenmodes": ["ChannelEigenmodes"],
    f"{base}.eigenmodes": ["eigenbasis"],
    f"{base}.initial_conditions": [
        "random_state", "random_vortical", "random_waves",
        "single_wave", "jet", "coherent_eddy"],
    f"{base}.model": ["Model"],
    f"{base}.modules.core": ["Core"],
    f"{base}.state": ["State"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
