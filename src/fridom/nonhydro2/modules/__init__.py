"""Nonhydrostatic modules.

Description
-----------
The concrete ``fr.Module`` subclasses of the nonhydrostatic model:
the dynamical core (declarations + pressure projection), the Coriolis
family, constant stratification, centered advection, and the wave
makers (Gaussian and polarized forcing).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        Relaxation,
    )

    from .advection import (
        CenteredAdvection,
        UpwindAdvection,
        WENOAdvection,
    )
    from .core import DynamicalCore
    from .gaussian_wave_maker import GaussianWaveMaker
    from .polarized_wave_maker import PolarizedWaveMaker
    from .pressure import SpectralPressureSolver
    from .smagorinsky_lilly import SmagorinskyLilly
    from .stratification import (
        ConstantStratification,
        MeridionalStratification,
    )

base = "fridom.nonhydro2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

# The Coriolis family and the generic relaxation are the shared
# framework module library (fr.modules), re-exported here so
# nh.modules.FPlaneCoriolis / nh.modules.Relaxation keep working
# after the wave-6 consolidation.
all_imports_by_origin = {
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "Relaxation"],
    f"{base}.core": ["DynamicalCore"],
    f"{base}.stratification": [
        "ConstantStratification", "MeridionalStratification"],
    f"{base}.advection": [
        "CenteredAdvection", "UpwindAdvection", "WENOAdvection"],
    f"{base}.pressure": ["SpectralPressureSolver"],
    f"{base}.gaussian_wave_maker": ["GaussianWaveMaker"],
    f"{base}.polarized_wave_maker": ["PolarizedWaveMaker"],
    f"{base}.smagorinsky_lilly": ["SmagorinskyLilly"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
