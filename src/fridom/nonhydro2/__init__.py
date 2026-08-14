"""``fridom.nonhydro2`` — the nonhydrostatic model.

Description
-----------
Phase-2 port of the nonhydrostatic model onto ``fridom``
(renamed onto ``fridom.nonhydro`` at cutover). Consumes the
framework (``fridom.spatial`` / ``fridom.model``) read-only. The
public surface mirrors the API sketches (§7):
``nh.Model`` (a preset factory), ``nh.State`` (the vocabulary class),
``nh.eigenmodes`` (the discrete-dispersion eigenmodes),
``nh.eigenbasis`` / ``nh.channel_eigenmodes`` (the labeled numeric
eigenmodes of the horizontally walled channel), and the concrete
modules (``nh.Core``, ``nh.FPlaneCoriolis``,
``nh.ConstantStratification``, ``nh.CenteredAdvection``, ...).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        CenteredAdvection,
        FPlaneCoriolis,
        RotationCoriolis,
        UpwindAdvection,
        WENOAdvection,
    )

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
    from .channel_eigenmodes import ChannelEigenmodes
    from .eigenmodes import eigenbasis
    from .initial_conditions import (
        barotropic_jet,
        coherent_eddy,
        eddy_dipole,
        flat_spectrum,
        gaussian,
        geostrophic_energy_spectrum,
        jet,
        kelvin_wave,
        random_state,
        random_vortical,
        random_waves,
        single_wave,
        wave_package,
    )
    from .model import Model
    from .modules.buoyancy_tracer import BuoyancyTracer
    from .modules.core import Core
    from .modules.smagorinsky_lilly import SmagorinskyLilly
    from .modules.stratification import (
        ConstantStratification,
        MeridionalStratification,
    )
    from .modules.thermal_wind import ThermalWindBackground
    from .state import State

base = "fridom.nonhydro2"

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
        "single_wave", "kelvin_wave", "wave_package",
        "gaussian", "barotropic_jet", "jet",
        "coherent_eddy", "eddy_dipole",
        # the spectrum vocabulary of ``spectral_energy_density=``
        "geostrophic_energy_spectrum", "flat_spectrum"],
    f"{base}.model": ["Model"],
    f"{base}.state": ["State"],
    f"{base}.modules.buoyancy_tracer": ["BuoyancyTracer"],
    f"{base}.modules.core": ["Core"],
    # the Coriolis and flux-form advection families are the shared
    # framework module library (advection rehomed under HY-D5)
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "RotationCoriolis",
        "CenteredAdvection", "UpwindAdvection", "WENOAdvection",
        "WindStress", "SurfaceBuoyancyFlux"],
    f"{base}.modules.stratification": [
        "ConstantStratification", "MeridionalStratification"],
    f"{base}.modules.smagorinsky_lilly": ["SmagorinskyLilly"],
    f"{base}.modules.thermal_wind": ["ThermalWindBackground"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
