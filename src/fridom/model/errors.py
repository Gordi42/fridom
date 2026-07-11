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

from fridom.spatial.errors import GridFrozenError
from fridom.spatial.operators.registry import (
    DispatchCollisionError,
)
from fridom.model.io.streams import (
    IOCollisionError,
    SnapshotMismatchError,
)
from fridom.model.results import (
    PanicError,
    RunTargetError,
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
