"""
Model-layer cluster of framework2.

Description
-----------
The re-export table below grows wave by wave with the Phase-2
implementation plan (``notes/framework2/model/implementation_plan.md``);
the class specs under ``notes/framework2/model/classes/`` are the
design contract. Names destined for the framework top level
(``fr.Model``, ``fr.Ramp``, ``fr.params``, ``fr.time_steppers``, ...)
are re-exported from ``fridom.framework2`` once their wave lands:

- Wave 2 adds the declaration vocabulary (declarations,
  space_patterns, roles, parameters, params, time_dependent, terms,
  implicit, stages, context).
- Wave 3 adds field_table, module, composer, schedule.
- Wave 4 adds model, assembly, report, results, clock, errors and
  the ``time_steppers`` namespace.
- Wave 6 adds ``closures``.
- Wave 7 adds term_predicates (``fr.terms``).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        assembly,
        composer,
        context,
        declarations,
        field_table,
        implicit,
        module,
        parameters,
        params,
        roles,
        schedule,
        space_patterns,
        stages,
        terms,
        time_dependent,
        time_steppers,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.model"

all_modules_by_origin = {
    base: [
        "time_steppers",
        "declarations",
        "space_patterns",
        "roles",
        "parameters",
        "params",
        "time_dependent",
        "terms",
        "implicit",
        "stages",
        "context",
        "field_table",
        "assembly",
        "module",
        "composer",
        "schedule",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
