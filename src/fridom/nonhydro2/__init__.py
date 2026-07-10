"""``fridom.nonhydro2`` — the nonhydrostatic model on framework2.

Description
-----------
Phase-2 port of the nonhydrostatic model onto ``fridom.framework2``
(renamed onto ``fridom.nonhydro`` at cutover). Consumes framework2
read-only. The public surface mirrors the API sketches (§7):
``nh.Model`` (a preset factory), ``nh.State`` (the vocabulary class),
``nh.eigenmodes`` (the discrete-dispersion eigenmodes),
``nh.eigenbasis`` / ``nh.channel_eigenmodes`` (the labeled numeric
eigenmodes of the horizontally walled channel), and the concrete
modules (``nh.DynamicalCore``, ``nh.FPlaneCoriolis``,
``nh.ConstantStratification``, ``nh.CenteredAdvection``, ...).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
    )

    from . import (
        channel_eigenmodes,
        diagnostics,
        eigenmodes,
        initial_conditions,
        modules,
        params,
        transforms,
    )
    from .channel_eigenmodes import ChannelEigenmodes
    from .eigenmodes import eigenbasis
    from .initial_conditions import (
        random_state,
        random_vortical,
        random_waves,
    )
    from .model import Model
    from .modules.advection import CenteredAdvection
    from .modules.core import DynamicalCore
    from .modules.stratification import (
        ConstantStratification,
        MeridionalStratification,
    )
    from .state import State

base = "fridom.nonhydro2"

all_modules_by_origin = {
    base: ["modules", "eigenmodes", "channel_eigenmodes",
           "diagnostics", "params", "transforms",
           "initial_conditions"],
}

all_imports_by_origin = {
    f"{base}.channel_eigenmodes": ["ChannelEigenmodes"],
    f"{base}.eigenmodes": ["eigenbasis"],
    f"{base}.initial_conditions": [
        "random_state", "random_vortical", "random_waves"],
    f"{base}.model": ["Model"],
    f"{base}.state": ["State"],
    f"{base}.modules.core": ["DynamicalCore"],
    # the Coriolis family is the shared framework module library
    "fridom.framework2.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis"],
    f"{base}.modules.stratification": [
        "ConstantStratification", "MeridionalStratification"],
    f"{base}.modules.advection": ["CenteredAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
