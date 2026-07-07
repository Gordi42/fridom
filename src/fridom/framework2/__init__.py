"""
Transitional parallel framework package for the grid redesign.

Description
-----------
Hosts the ``framework2.grid`` package and the Phase-2 model layer
(``model``, ``transforms``, ``io``, ``ops``), all renamed to
``fridom.framework.*`` at cutover; see
``notes/framework2/classes/README.md`` and
``notes/framework2/model/classes/README.md`` for the design
contracts. ``framework2`` may import ``fridom.framework.utils`` but
must not import the old model/grid stack.

Top-level re-exports (``fr.Model``, ``fr.Ramp``, ``fr.params``,
``fr.StateTransform``, ``fr.every``/``fr.at``, ...) are added wave by
wave with the Phase-2 implementation plan
(``notes/framework2/model/implementation_plan.md``).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import grid, io, model, modules, ops, transforms
    from .io import slurm

    # import all classes (the fr.* surface, grown wave by wave)
    from .io.triggers import at, every
    from .model import implicit, params, roles, time_steppers
    from .model.clock import Clock
    from .model.context import StepContext
    from .model.declarations import (
        FieldDeclaration,
        FieldReference,
        Lifecycle,
    )
    from .model.model import Model
    from .model.module import Module
    from .model.parameters import (
        USE_PROVIDED,
        Param,
        ParameterDeclaration,
        ParameterReference,
    )
    from .model.results import (
        AdvanceResult,
        PanicError,
        RunResult,
        RunStatus,
        RunTargetError,
    )
    from .model.space_patterns import (
        Collocated,
        Dof,
        Profile,
        SpacePattern,
        SpaceRule,
        Staggered,
    )
    from .model.stages import Stage, StageKind, self_update
    from .model.terms import (
        EXPLICIT,
        IMPLICIT,
        TendencyTerm,
        Treatment,
        term,
    )
    from .model.time_dependent import Ramp, TimeDependent, resolve_at

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2"

all_modules_by_origin = {
    base: ["grid", "model", "modules", "transforms", "io", "ops"],
    # module namespaces re-homed to the top level (fr.roles, ...)
    f"{base}.model": ["roles", "params", "implicit", "time_steppers"],
    f"{base}.io": ["slurm"],
}

all_imports_by_origin = {
    f"{base}.model.declarations": [
        "FieldDeclaration", "Lifecycle", "FieldReference"],
    f"{base}.model.space_patterns": [
        "Dof", "SpacePattern", "Collocated", "Staggered", "Profile",
        "SpaceRule"],
    f"{base}.model.parameters": [
        "ParameterDeclaration", "ParameterReference", "Param",
        "USE_PROVIDED"],
    f"{base}.model.time_dependent": [
        "TimeDependent", "Ramp", "resolve_at"],
    f"{base}.model.terms": [
        "TendencyTerm", "term", "Treatment", "EXPLICIT", "IMPLICIT"],
    f"{base}.model.stages": ["Stage", "StageKind", "self_update"],
    f"{base}.model.context": ["StepContext"],
    f"{base}.model.module": ["Module"],
    f"{base}.model.model": ["Model"],
    f"{base}.model.clock": ["Clock"],
    f"{base}.model.results": [
        "RunStatus", "AdvanceResult", "RunResult", "PanicError",
        "RunTargetError"],
    f"{base}.io.triggers": ["every", "at"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
