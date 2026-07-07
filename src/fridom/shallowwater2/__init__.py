"""
The shallow-water model, ported onto framework2.

Description
-----------
A top-level parallel package (renamed onto ``fridom.shallowwater`` at
cutover) that consumes ``fridom.framework2`` and provides the
shallow-water vocabulary and physics:

- :class:`~fridom.shallowwater2.state.State` — the ``u``/``v``/``p``
  vocabulary class;
- :mod:`~fridom.shallowwater2.modules` — the core, Coriolis, and
  Sadourny advection modules;
- :func:`~fridom.shallowwater2.model.Model` — the thin preset factory
  (``sw.Model(grid=..., coriolis=..., ...)``);
- :mod:`~fridom.shallowwater2.eigenmodes` — ``from_model`` and the
  linear dispersion eigenmodes;
- :mod:`~fridom.shallowwater2.params` — the package parameter names.

Lazy re-exports (mirroring ``fridom.framework2.grid.__init__``): the
framework surface itself stays under ``fridom.framework2`` (imported
as ``fr``).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import eigenmodes, modules, params

    # import all classes
    from .model import Model
    from .state import State

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.shallowwater2"

all_modules_by_origin = {
    base: ["modules", "eigenmodes", "params"],
}

all_imports_by_origin = {
    f"{base}.model": ["Model"],
    f"{base}.state": ["State"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
