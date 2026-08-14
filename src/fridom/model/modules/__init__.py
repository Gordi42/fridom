"""
The framework's shared tendency-module library.

Description
-----------
Reusable, model-agnostic ``fr.model.Module`` subclasses shared across the
model ports (D2.1 module-library sharing). Wave 6 seeds it with the
Coriolis family (``fr.model.modules.FPlaneCoriolis`` /
``fr.model.modules.BetaPlaneCoriolis``); both the nonhydrostatic and
shallow-water packages import from here instead of carrying a copy.
The forcing port adds the generic ``fr.model.modules.Relaxation``. The
flux-form advection family (``fr.model.modules.CenteredAdvection`` /
``UpwindAdvection`` / ``WENOAdvection``) is rehomed here from
``nonhydro2`` so every model port shares it (hydrostatic plan HY-D5);
``nonhydro2`` keeps re-exports so ``nh.CenteredAdvection`` still works.
``fr.model.modules.Tracer`` is the one-liner declaring module for plain
user tracers (D1.5), the replacement for the v1
``mset.custom_state_fields``. The ocean surface-forcing wrappers
(``fr.model.modules.WindStress`` / ``SurfaceBuoyancyFlux``) are rehomed
here from ``nonhydro2`` for the same reason as advection — hydrostatic
needs the identical oceanographic sign convention, and one shared
wrapper cannot drift into two; ``nonhydro2`` keeps re-exports so
``nh.WindStress`` still works.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .advection import (
        CenteredAdvection,
        UpwindAdvection,
        WENOAdvection,
    )
    from .boundary_flux import BoundaryFlux
    from .coriolis import (
        OMEGA_EARTH,
        RADIUS_EARTH,
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        RotationCoriolis,
        chart_rotation,
        linear_rotation,
    )
    from .immersed import MaskState
    from .moving_geometry import (
        MeshVelocityCorrection,
        MovingGeometry,
        mapping_params,
    )
    from .ramping import TendencyEnvelope
    from .relaxation import Relaxation
    from .source import Source
    from .surface_forcing import SurfaceBuoyancyFlux, WindStress
    from .tracer import Tracer

base = "fridom.model.modules"

all_modules_by_origin: dict[str, list[str]] = {}

all_imports_by_origin = {
    f"{base}.advection": ["CenteredAdvection", "UpwindAdvection",
                          "WENOAdvection"],
    f"{base}.boundary_flux": ["BoundaryFlux"],
    f"{base}.coriolis": ["FPlaneCoriolis", "BetaPlaneCoriolis",
                          "RotationCoriolis", "linear_rotation",
                          "chart_rotation", "OMEGA_EARTH",
                          "RADIUS_EARTH"],
    f"{base}.immersed": ["MaskState"],
    f"{base}.moving_geometry": ["MovingGeometry",
                                "MeshVelocityCorrection",
                                "mapping_params"],
    f"{base}.ramping": ["TendencyEnvelope"],
    f"{base}.relaxation": ["Relaxation"],
    f"{base}.source": ["Source"],
    f"{base}.surface_forcing": ["WindStress", "SurfaceBuoyancyFlux"],
    f"{base}.tracer": ["Tracer"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
