"""
``fridom.model`` — running a model through time.

Description
-----------
Everything about orchestrating a model through time: the model core
(assembly, run loop, schedule), tendency modules, time steppers,
state transforms, io, and ops. Consumes ``fridom.spatial`` read-only;
the concrete models (``fridom.nonhydro2``, ``fridom.shallowwater2``)
build on this package.
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
        clock,
        closures,
        composer,
        context,
        declarations,
        eigen,
        eigen_channel,
        errors,
        field_table,
        implicit,
        io,
        module,
        modules,
        ops,
        parameters,
        params,
        report,
        results,
        roles,
        schedule,
        stages,
        term_predicates,
        terms,
        time_dependent,
        time_steppers,
        transforms,
    )
    from ._eigenbasis import eigenbasis

    # import all classes (the fr.model.* surface)
    from .clock import Clock
    from .context import StepContext
    from .declarations import (
        FieldDeclaration,
        FieldReference,
        Lifecycle,
    )
    from .eigen import NumericEigenmodes, numeric_eigenpairs
    from .eigen_channel import ChannelEigenbasis, channel_eigenpairs
    from .energy import EnergyMetric
    from .io import slurm
    from .io.triggers import at, every
    from .model import Model
    from .module import Module
    from .parameters import (
        USE_PROVIDED,
        Param,
        ParameterDeclaration,
        ParameterReference,
        leaf,
    )
    from .results import (
        AdvanceResult,
        PanicError,
        RunResult,
        RunStatus,
        RunTargetError,
    )
    from .stages import Stage, StageKind, self_update
    from .term_predicates import (
        linear_operator_gaps,
        linearize,
        require_linear_operator,
    )
    from .terms import (
        EXPLICIT,
        IMPLICIT,
        TendencyTerm,
        Treatment,
        term,
    )
    from .time_dependent import Ramp, TimeDependent, resolve_at
    from .transforms.adiabatic_ramping import AdiabaticRamping
    from .transforms.base import StateTransform
    from .transforms.optimal_balance import OptimalBalance
    from .transforms.propagator import Propagator
    from .transforms.time_average import TimeAverage

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.model"

all_modules_by_origin = {
    base: [
        "time_steppers", "closures", "declarations", "roles",
        "parameters", "params", "time_dependent", "terms",
        "term_predicates", "implicit", "stages", "context",
        "field_table", "assembly", "module", "composer", "schedule",
        "clock", "results", "report", "errors", "eigen",
        "eigen_channel", "modules", "transforms", "io", "ops"],
    f"{base}.io": ["slurm"],
}

all_imports_by_origin = {
    f"{base}.declarations": [
        "FieldDeclaration", "Lifecycle", "FieldReference"],
    f"{base}.parameters": [
        "ParameterDeclaration", "ParameterReference", "Param",
        "USE_PROVIDED", "leaf"],
    f"{base}.time_dependent": [
        "TimeDependent", "Ramp", "resolve_at"],
    f"{base}.terms": [
        "TendencyTerm", "term", "Treatment", "EXPLICIT", "IMPLICIT"],
    f"{base}.stages": ["Stage", "StageKind", "self_update"],
    f"{base}.context": ["StepContext"],
    f"{base}.module": ["Module"],
    f"{base}.model": ["Model"],
    f"{base}.clock": ["Clock"],
    f"{base}.eigen": ["NumericEigenmodes", "numeric_eigenpairs"],
    f"{base}.eigen_channel": [
        "ChannelEigenbasis", "channel_eigenpairs"],
    f"{base}._eigenbasis": ["eigenbasis"],
    f"{base}.energy": ["EnergyMetric"],
    f"{base}.term_predicates": [
        "linearize", "require_linear_operator",
        "linear_operator_gaps"],
    f"{base}.transforms.base": ["StateTransform"],
    f"{base}.transforms.propagator": ["Propagator"],
    f"{base}.transforms.time_average": ["TimeAverage"],
    f"{base}.transforms.adiabatic_ramping": ["AdiabaticRamping"],
    f"{base}.transforms.optimal_balance": ["OptimalBalance"],
    f"{base}.results": [
        "RunStatus", "AdvanceResult", "RunResult", "PanicError",
        "RunTargetError"],
    f"{base}.io.triggers": ["every", "at"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
