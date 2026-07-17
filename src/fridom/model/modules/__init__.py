"""
The framework's shared tendency-module library.

Description
-----------
Reusable, model-agnostic ``fr.Module`` subclasses shared across the
model ports (D2.1 module-library sharing). Wave 6 seeds it with the
Coriolis family (``fr.modules.FPlaneCoriolis`` /
``fr.modules.BetaPlaneCoriolis``); both the nonhydrostatic and
shallow-water packages import from here instead of carrying a copy.
The forcing port adds the generic ``fr.modules.Relaxation``. The
flux-form advection family (``fr.modules.CenteredAdvection`` /
``UpwindAdvection`` / ``WENOAdvection``) is rehomed here from
``nonhydro2`` so every model port shares it (hydrostatic plan HY-D5);
``nonhydro2`` keeps re-exports so ``nh.CenteredAdvection`` still works.
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
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        RotationCoriolis,
        chart_rotation,
        linear_rotation,
    )
    from .moving_geometry import (
        MeshVelocityCorrection,
        MovingGeometry,
        mapping_params,
    )
    from .relaxation import Relaxation

base = "fridom.model.modules"

all_modules_by_origin: dict[str, list[str]] = {}

all_imports_by_origin = {
    f"{base}.advection": ["CenteredAdvection", "UpwindAdvection",
                          "WENOAdvection"],
    f"{base}.boundary_flux": ["BoundaryFlux"],
    f"{base}.coriolis": ["FPlaneCoriolis", "BetaPlaneCoriolis",
                          "RotationCoriolis", "linear_rotation",
                          "chart_rotation"],
    f"{base}.moving_geometry": ["MovingGeometry",
                                "MeshVelocityCorrection",
                                "mapping_params"],
    f"{base}.relaxation": ["Relaxation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
