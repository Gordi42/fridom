"""``fridom.nonhydro2`` — the nonhydrostatic model on framework2.

Description
-----------
Phase-2 port of the nonhydrostatic model onto ``fridom.framework2``
(renamed onto ``fridom.nonhydro`` at cutover). Consumes framework2
read-only. The public surface mirrors the API sketches (§7):
``nh.Model`` (a preset factory), ``nh.State`` (the vocabulary class),
``nh.eigenmodes`` (the discrete-dispersion eigenmodes), and the
concrete modules (``nh.DynamicalCore``, ``nh.FPlaneCoriolis``,
``nh.ConstantStratification``, ``nh.CenteredAdvection``, ...).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from . import diagnostics, eigenmodes, modules, params
    from .model import Model
    from .modules.advection import CenteredAdvection
    from .modules.core import DynamicalCore
    from .modules.coriolis import BetaPlaneCoriolis, FPlaneCoriolis
    from .modules.stratification import ConstantStratification
    from .state import State

base = "fridom.nonhydro2"

all_modules_by_origin = {
    base: ["modules", "eigenmodes", "diagnostics", "params"],
}

all_imports_by_origin = {
    f"{base}.model": ["Model"],
    f"{base}.state": ["State"],
    f"{base}.modules.core": ["DynamicalCore"],
    f"{base}.modules.coriolis": [
        "FPlaneCoriolis", "BetaPlaneCoriolis"],
    f"{base}.modules.stratification": ["ConstantStratification"],
    f"{base}.modules.advection": ["CenteredAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
