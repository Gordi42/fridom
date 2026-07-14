"""
The model-layer error registry.

Description
-----------
The concrete model-layer errors of
``design/specs/model/classes/model.md`` section 7. Assembly-time
errors subclass the common ``AssemblyError``; teaching shims
subclass ``TypeError``-flavored bases so they read as API misuse.
Errors owned by other clusters (``GridFrozenError``,
``MissingComponentError``, ``SignatureMismatchError``,
``IOCollisionError``, ``SnapshotMismatchError``, ``PanicError``,
``RunTargetError``) live at their raise sites and are only
re-exported or re-raised here. ``DispatchCollisionError`` is defined
with the operator registry (its raise site is the merge call) and
re-exported for the assembly surface.

Shared file: orchestrator-owned; raise sites land wave by wave.
"""
from __future__ import annotations

from fridom.model.io.streams import (
    IOCollisionError,
    SnapshotMismatchError,
)
from fridom.model.results import (
    PanicError,
    RunTargetError,
)
from fridom.spatial.errors import GridFrozenError
from fridom.spatial.operators.registry import (
    DispatchCollisionError,
)

__all__ = [
    "AssemblyError",
    "DispatchCollisionError",
    "FieldCollisionError",
    "GridFrozenError",
    "IOCollisionError",
    "ImmutableParameterError",
    "ImmutableStateError",
    "ImplicitCollisionError",
    "LinearTermInTendencyError",
    "MissingFieldError",
    "MissingParameterError",
    "PanicError",
    "ParameterCollisionError",
    "RunTargetError",
    "SnapshotMismatchError",
    "TermEvaluationError",
    "TimeDependentParameterError",
]


class AssemblyError(ValueError):

    """
    Base class of the assembly-time errors.

    Description
    -----------
    Raised while ``fr.Model`` runs the nine-step assembly pipeline;
    every subclass attributes the failure to the responsible
    module(s) in its message. Never raised in-trace.
    """


class FieldCollisionError(AssemblyError):

    """
    Raised when two modules declare the same field name.

    Description
    -----------
    Assembly step 1: two ``FieldDeclaration``s of one state
    component; the message names **both** modules — declarations are
    never silently merged.
    """


class MissingFieldError(AssemblyError):

    """
    Raised when a ``FieldReference`` is unsatisfied.

    Description
    -----------
    Assembly step 1: a module references a field no other module
    declares; the message carries the reference's hint.
    """


class MissingParameterError(AssemblyError):

    """
    Raised when a REQUIRED parameter reference is unsatisfied.

    Description
    -----------
    Assembly step 2 (a ``ParameterReference`` with no provider and
    no default), the host ``model.parameters[...]`` lookup, and the
    lazy bound-diagnostic call. Attributed to the requiring
    module/diagnostic; the message adds the ``fr.params`` registry
    hint and the list of provided parameters.
    """


class ParameterCollisionError(AssemblyError):

    """
    Raised when two modules provide the same dotted parameter name.

    Description
    -----------
    Assembly step 2: one-provider-per-name; the message names both
    modules.
    """


class ImplicitCollisionError(AssemblyError):

    """
    Raised on more than one non-mergeable implicit operator per field.

    Description
    -----------
    Assembly step 5: mergeable implicit families sum exactly; a
    second non-mergeable custom operator on one field is an error
    naming both terms.
    """


class TimeDependentParameterError(AssemblyError):

    """
    Raised on a bind-time read of a time-dependent parameter.

    Description
    -----------
    ``bind`` (and ``from_model`` eigenmode/transform constructors)
    must spell time-dependent reads ``at_time(0.0)`` /
    ``at_time=``; a bare read would freeze a stale coefficient. The
    message teaches the split: grid factors resolve at bind,
    parameter factors resolve in-step.
    """


class LinearTermInTendencyError(AssemblyError):

    """
    Raised when an exponential stepper meets a linear tendency term.

    Description
    -----------
    The exponential (ETD) family supplies the linear operator itself
    — ``exp(L dt)`` comes from the model's eigenbasis, not from the
    tendency. A model that ALSO evaluates its ``linear=True`` terms
    in the tendency therefore integrates them **twice**: silently
    wrong physics, never a crash. Raised host-side at trace time (the
    schedule is static), so it fires on the first compile rather than
    in the science.

    The fix is the model contract: assemble with the linear terms
    filtered out, and hand the stepper the eigenbasis of the
    UNFILTERED model.

    Parameters
    ----------
    terms : tuple[str, ...]
        The offending ``"Module/term"`` attribution keys.
    """

    def __init__(self, terms: tuple[str, ...]) -> None:
        """Compose the taught double-counting message."""
        self.terms = tuple(terms)
        listed = ", ".join(self.terms)
        super().__init__(
            f"the tendency still carries linear terms ({listed}), but "
            "an exponential stepper integrates the linear operator "
            "itself through exp(L dt) — they would be counted twice. "
            "Assemble the model with the linear terms filtered out:\n"
            "    from fridom.model import term_predicates as terms\n"
            "    model = sw.Model(..., term_filter=~terms.linear,\n"
            "                     time_stepper=fr.model.time_steppers"
            ".ETDRK4(dt, basis))\n"
            "and build `basis` from the UNFILTERED model "
            "(sw.eigenbasis(full_model)), so exp(L dt) is the "
            "operator those terms describe.")


class LinearOperatorGapError(ValueError):

    """
    Raised when a consumer of ``L`` meets a module that is not in it.

    Description
    -----------
    Some modules carry physics that *belongs* in the linear operator
    but is declared inside a ``linear=False`` term — the shallow-water
    conserving Coriolis family (``sw.modules.NonlinearFPlaneCoriolis``
    and friends) is the motivating case: it carries rotation whole,
    inside its nonlinear thickness-ratio term, so
    ``fr.model.linearize`` would build a **non-rotating** ``L``.
    Eigenmodes, projections, optimal balance and IMEX-by-linearity
    built on such an ``L`` are not degraded, they are *wrong*.

    Such a module declares the gap on itself
    (``Module.linear_operator_gap``, a sentence saying what is missing
    and what to do instead); every consumer of ``L`` calls
    ``fr.model.require_linear_operator(model, consumer=...)`` first and
    raises this rather than silently handing out a rotation-free
    operator.
    """


class TermEvaluationError(RuntimeError):

    """
    Wraps an exception raised inside a term or stage.

    Description
    -----------
    The composer's trace-time wrapper (and the assembly dry run):
    the original exception is chained, the message carries the
    ``"Module/term"`` attribution key; grid-layer space mismatches
    are re-raised with the same attribution.
    """


class ImmutableStateError(TypeError):

    """
    Raised by ``State``/``VectorField`` mutation attempts.

    Description
    -----------
    Teaching shim on property setters and ``__setitem__``: state
    updates are functional — the guidance points at ``replace`` and
    ``add``.
    """


class ImmutableParameterError(TypeError):

    """
    Raised on post-assembly module attribute pokes.

    Description
    -----------
    Teaching shim: assembled modules are read-only from the host —
    the guidance points at ``update_parameters`` (leaf values) or a
    re-assembly (structure).
    """
