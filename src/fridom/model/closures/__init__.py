"""
The closure namespace (``fr.closures``).

Description
-----------
Owning class spec: ``design/specs/model/classes/module.md``.
``ClosureBase`` hosts the role-target resolution (D1.4, V-H2) and is
the ``fr.terms.owned_by`` predicate target; the concrete diffusion
family (harmonic/biharmonic mixing and friction) ports the old
framework closures onto role-driven targets and provided-parameter
coefficients.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all classes
    from .base import ClosureBase
    from .diffusion import (
        BiharmonicDiffusion,
        BiharmonicFriction,
        HarmonicDiffusion,
        HarmonicFriction,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.model.closures"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base}.base": ["ClosureBase"],
    f"{base}.diffusion": [
        "HarmonicDiffusion", "BiharmonicDiffusion",
        "HarmonicFriction", "BiharmonicFriction"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
