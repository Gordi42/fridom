"""
The model-layer error registry.

Description
-----------
The concrete model-layer errors of
``design/specs/model/classes/model.md`` section 7. Assembly-time
errors subclass the common ``AssemblyError``; teaching shims
subclass ``TypeError``-flavored bases so they read as API misuse.
Errors owned by other clusters (``GridFrozenError``,
``ImmutableStateError``, ``MissingComponentError``,
``SignatureMismatchError``, ``IOCollisionError``,
``SnapshotMismatchError``, ``PanicError``, ``RunTargetError``) live at
their raise sites and are only re-exported or re-raised here.
``ImmutableStateError`` moved to the grid cluster
(``fridom.spatial.errors``) with the raising ``ScalarField.data``
setter. ``DispatchCollisionError`` is defined
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
from fridom.spatial.errors import GridFrozenError, ImmutableStateError
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
    "IrreversibleTermError",
    "LinearTermInTendencyError",
    "MissingFieldError",
    "MissingParameterError",
    "PanicError",
    "ParameterCollisionError",
    "RunTargetError",
    "SnapshotMismatchError",
    "TermEvaluationError",
    "TimeDependentLinearOperatorError",
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


class TimeDependentLinearOperatorError(AssemblyError):

    """
    Raised when a frozen-``L`` stepper meets a time-dependent ``L``.

    Description
    -----------
    An exponential (ETD) stepper diagonalizes the LINEAR operator
    ``L`` once, host-side, and integrates it as ``exp(L dt)`` from a
    frozen eigenbasis snapshot. If a ``linear=True`` term consumes a
    time-dependent parameter (an ``fr.Ramp`` on ``coriolis.f0``, say),
    then ``L`` is really ``L(t)`` — and ``L(t_1)`` and ``L(t_2)`` do
    not commute, so ``exp(L dt)`` stops being the propagator and the
    scheme silently integrates a stale operator. Raised host-side at
    assembly (never in the science), so it fires on the first build.

    The fix is either the ``AdamBashforth`` fallback (which re-reads
    the tendency, hence every parameter, at each stage clock time), or
    the measured split: keep the stiff, time-INDEPENDENT part in the
    eigenbasis and leave the time-dependent term in the tendency (see
    ``design/research/exponential_stepper.md`` section 5).

    Parameters
    ----------
    offenders : tuple[tuple[str, str], ...]
        The offending ``(module type name, parameter name)`` pairs.
    stepper : str
        The exponential stepper's class name (named in the message).
    """

    def __init__(
        self, offenders: tuple[tuple[str, str], ...], stepper: str,
    ) -> None:
        """Compose the taught time-dependent-``L`` message."""
        self.offenders = tuple(offenders)
        self.stepper = stepper
        listed = ", ".join(
            f"{param} ({module})" for module, param in self.offenders)
        super().__init__(
            f"{stepper} freezes the linear operator L in an eigenbasis "
            "(it integrates exp(L dt) from a snapshot), but these "
            "time-dependent parameters feed linear=True terms that L "
            f"integrates: {listed}. L(t1) and L(t2) do not commute, so "
            "exp(L dt) is no longer the propagator and the ramp would "
            "be silently frozen. Either step with "
            "fr.model.time_steppers.AdamBashforth (it re-reads the "
            "tendency, hence every parameter, at each stage clock "
            "time), or keep the stiff time-INDEPENDENT part in the "
            "eigenbasis and leave the time-dependent term in the "
            "tendency (the measured split) — see "
            "design/research/exponential_stepper.md section 5.")


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


class IrreversibleTermError(ValueError):

    """
    Raised when a backward ramp leg retains an irreversible term.

    Description
    -----------
    The AR-D6 guard (``design/specs/model/08_state_transforms.md``
    §10.9 law 3). A backward ramp leg integrates the model with a
    negated ``fr.params.TIME_STEP``; backward diffusion is ill-posed,
    so an :class:`AdiabaticRamping` ``dt < 0`` leg refuses to build
    while its variant still carries terms matching
    ``fr.terms.owned_by(fr.closures.ClosureBase) | fr.terms.implicit``
    (dissipative closures and implicitly treated terms). Raised
    host-side at leg construction (never in the science), naming the
    offending terms; the fix is an explicit ``term_filter`` that drops
    them from the backward leg. Sign-reversed ramped dissipation is
    out of scope (a first-class ``reversible`` term tag may later
    replace the predicate heuristic).

    Parameters
    ----------
    terms : tuple[str, ...]
        The offending ``"Module/term"`` attribution keys.
    """

    def __init__(self, terms: tuple[str, ...]) -> None:
        """Compose the taught backward-irreversibility message."""
        self.terms = tuple(terms)
        listed = ", ".join(self.terms)
        super().__init__(
            f"a backward ramp leg (dt < 0) still carries irreversible "
            f"terms ({listed}): dissipative closures "
            "(fr.terms.owned_by(fr.closures.ClosureBase)) and "
            "implicitly treated terms (fr.terms.implicit) are "
            "ill-posed integrated backward in time. Drop them from "
            "the leg with an explicit term_filter, e.g.\n"
            "    from fridom.model import term_predicates as terms\n"
            "    leg = fr.transforms.AdiabaticRamping(\n"
            "        model, ramps=..., ramp_period=...,\n"
            "        term_filter=~terms.owned_by(fr.closures.ClosureBase)"
            " & ~terms.implicit)\n"
            "and take its .backward; sign-reversed ramped dissipation "
            "is out of scope.")


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


class ImmutableParameterError(TypeError):

    """
    Raised on post-assembly module attribute pokes.

    Description
    -----------
    Teaching shim: assembled modules are read-only from the host —
    the guidance points at ``update_parameters`` (leaf values) or a
    re-assembly (structure).
    """
