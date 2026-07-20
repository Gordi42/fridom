"""``fridom.hydrostatic`` — the hydrostatic primitive-equation model.

Description
-----------
The hydrostatic model on the new stack (ROADMAP 3.1), built like
``fridom.nonhydro2`` on ``fridom.spatial`` / ``fridom.model``
(read-only). Prognostic ``u, v`` (C-grid faces), buoyancy ``b``
(linear-EOS tracer) and surface pressure ``ps = g*eta`` (2D,
constant-along-z); diagnosed ``w`` (continuity) and ``p_hyd``
(hydrostatic balance), both recomputed in the S1' DIAGNOSE stages.

The public surface mirrors the sibling packages:
``hy.Model`` (a preset factory), ``hy.State`` (the vocabulary class),
and the concrete modules (``hy.HydrostaticCore``,
``hy.ConstantStratification``, ``hy.ExplicitFreeSurface``,
``hy.FPlaneCoriolis``, ...). ``hy.eigenmodes`` / ``hy.transforms``
are the numeric eigenbasis and its vortical / wave / barotropic /
baroclinic projections (HY-D7, stage H4).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.closures import VerticalMixing
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        RotationCoriolis,
    )

    from . import (
        comparison,
        diagnostics,
        eigenmodes,
        energy,
        initial_conditions,
        modules,
        params,
        transforms,
    )
    from .comparison import comparison_model
    from .eigenmodes import eigenbasis
    from .initial_conditions import jet, single_wave
    from .model import Model
    from .modules.core import HydrostaticCore
    from .modules.free_surface import (
        ExplicitFreeSurface,
        ImplicitFreeSurface,
        SplitExplicitFreeSurface,
    )
    from .modules.stratification import ConstantStratification
    from .modules.thermal_wind import ThermalWindBackground
    from .state import State

base = "fridom.hydrostatic"

all_modules_by_origin = {
    base: ["modules", "diagnostics", "energy", "params",
           "initial_conditions", "comparison", "eigenmodes",
           "transforms"],
}

all_imports_by_origin = {
    f"{base}.eigenmodes": ["eigenbasis"],
    f"{base}.initial_conditions": ["single_wave", "jet"],
    f"{base}.comparison": ["comparison_model"],
    f"{base}.model": ["Model"],
    f"{base}.state": ["State"],
    f"{base}.modules.core": ["HydrostaticCore"],
    f"{base}.modules.stratification": ["ConstantStratification"],
    f"{base}.modules.thermal_wind": ["ThermalWindBackground"],
    f"{base}.modules.free_surface": [
        "ExplicitFreeSurface", "ImplicitFreeSurface",
        "SplitExplicitFreeSurface"],
    # the Coriolis family and the implicit vertical mixing closure are
    # shared framework module libraries, re-exported here
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "RotationCoriolis"],
    "fridom.model.closures": ["VerticalMixing"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
