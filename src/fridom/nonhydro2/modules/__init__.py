"""Nonhydrostatic modules.

Description
-----------
The concrete ``fr.model.Module`` subclasses of the nonhydrostatic model:
the dynamical core (declarations + pressure projection), the Coriolis
family, the buoyancy family (constant/meridional stratification and
the bare buoyancy tracer), and centered advection. Volumetric forcing
(the former Gaussian / polarized wave makers) is the model-agnostic
``fr.model.modules.Source``, fed a wave-packet pattern by
``nh.initial_conditions.wave_package``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        CenteredAdvection,
        FPlaneCoriolis,
        Relaxation,
        RotationCoriolis,
        SurfaceBuoyancyFlux,
        UpwindAdvection,
        WENOAdvection,
        WindStress,
    )

    from .buoyancy_tracer import BuoyancyTracer
    from .composed_pressure import ComposedPressureSolver
    from .core import Core
    from .immersed_pressure import ImmersedPressureSolver
    from .mapped_pressure import MappedPressureSolver
    from .pressure import SpectralPressureSolver
    from .smagorinsky_lilly import SmagorinskyLilly
    from .stratification import (
        ConstantStratification,
        MeridionalStratification,
    )
    from .thermal_wind import ThermalWindBackground

base = "fridom.nonhydro2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

# The Coriolis family, the flux-form advection family (rehomed under
# HY-D5), the generic relaxation, and the ocean surface-forcing wrappers
# are the shared framework module library (fr.model.modules), re-exported
# here so nh.modules.FPlaneCoriolis / nh.modules.CenteredAdvection /
# nh.modules.Relaxation / nh.modules.WindStress keep working after the
# consolidation.
all_imports_by_origin = {
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis",
        "RotationCoriolis", "Relaxation",
        "CenteredAdvection", "UpwindAdvection", "WENOAdvection",
        "WindStress", "SurfaceBuoyancyFlux"],
    f"{base}.buoyancy_tracer": ["BuoyancyTracer"],
    f"{base}.core": ["Core"],
    f"{base}.stratification": [
        "ConstantStratification", "MeridionalStratification"],
    f"{base}.pressure": ["SpectralPressureSolver"],
    f"{base}.mapped_pressure": ["MappedPressureSolver"],
    f"{base}.immersed_pressure": ["ImmersedPressureSolver"],
    f"{base}.composed_pressure": ["ComposedPressureSolver"],
    f"{base}.smagorinsky_lilly": ["SmagorinskyLilly"],
    f"{base}.thermal_wind": ["ThermalWindBackground"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
