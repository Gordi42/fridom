"""
Stage declarations.

Description
-----------
``StageKind`` (the closed schedule vocabulary), ``Stage`` (a
module-owned stage declaration), and the ``fr.self_update``
decorator. Owning class spec:
``design/specs/model/classes/module.md``; design source
``design/specs/model/03_time_stepping.md`` sections 5.2/5.5.

Stages are transient assembly inputs mirroring ``TendencyTerm``:
plain frozen host objects consumed at assembly and discarded; the
composed step closes over slot indices and unbound functions.
Schedule position is a pure function of the declared *kind*, never
of module list position; ordering, write-gate validation, and the
overlapping-writes lint are assembly machinery (wave 3+).
"""
# Wave 2 C: StageKind, Stage, self_update
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable


# attribute under which @fr.self_update stamps the Stage declaration
# onto the method; assembly (wave 3+) wraps either spelling — the
# stamped method or the bare undecorated method — into the schedule
STAGE_ATTRIBUTE: Final[str] = "__fridom_stage__"

# module-private sentinel: cadence= is RESERVED (CS-1) — the keyword
# is accepted only to be rejected, so even an explicit None raises
_CADENCE_RESERVED: Final = object()


# ================================================================
#  StageKind
# ================================================================
class StageKind(enum.Enum):

    """
    The closed iteration-1 stage vocabulary.

    Description
    -----------
    Schedule slot is a pure function of the kind (the canonical
    step, 03 section 5.2)::

        per substage:  P0 ctx -> S1 SELF_UPDATE -> S1' DIAGNOSE
                       -> S2 terms -> S3 primary ADVANCE (stepper)
                       -> S3' ADVANCE stages -> S4 CONSTRAINT
        per step:      S5 NaN seam -> S6 DIAGNOSTIC -> clock tick

    The kind set is CLOSED: no open kind vocabulary and no
    "insert before X" API in iteration 1 — user extension is kind
    declaration (positivity clamp -> CONSTRAINT, step-cadence
    accumulator -> DIAGNOSTIC, time-dependent geometry ->
    SELF_UPDATE). Tendency contributions are terms, not stages; the
    primary advance (S3) is the stepper's, not a module stage.
    """

    SELF_UPDATE = enum.auto()   # S1,  per substage
    DIAGNOSE = enum.auto()      # S1', per substage
    ADVANCE = enum.auto()       # S3', per substage
    CONSTRAINT = enum.auto()    # S4,  per substage
    DIAGNOSTIC = enum.auto()    # S6,  per step (post-NaN)


# ================================================================
#  Stage
# ================================================================
@dataclass(frozen=True)
class Stage:

    """
    A module-owned stage declaration (transient assembly data).

    Description
    -----------
    Mirrors ``TendencyTerm``: a frozen host record consumed at
    assembly, never a pytree. The body is an arbitrary pure function
    ``(module, state, ctx) -> dict`` applied per the kind's write
    gate (``replace``; terms: ``add``); ``fn`` is stored UNBOUND and
    paired with a module slot at compose time (the aliasing rule).

    ``cadence=`` is deliberately NOT a field: it is reserved on the
    ``self_update`` declaration (CS-1) and not built.

    Parameters
    ----------
    kind : StageKind
        The declared kind; schedule position is a pure function of
        it (D1.3 commitment 5).
    fn : Callable | str
        UNBOUND ``(module, state, ctx) -> dict``; a string is the
        method name, resolved to the unbound method at collection.
    name : str | None
        Attribution key part (``"Module/stage"``); defaults to the
        ``fn`` name (default: None).
    order : int
        Explicit intra-kind order; ties broken by (module tuple
        index, declaration index) — an assembly lint errors on
        same-kind stages with overlapping write sets and equal
        order (default: 0).
    advances : tuple[str, ...]
        ADVANCE or CONSTRAINT only: the named PROGNOSTIC subset this
        stage advances; counts as "advanced" in the coverage lint
        (spec 5.4 lint amendment). A CONSTRAINT stage claims only
        the fields it genuinely integrates forward (the implicit
        free surface claims its ``ps``), not its whole write set —
        a projection that merely corrects velocities claims nothing
        (default: ()).
    reads : tuple[str, ...]
        SELF_UPDATE only: state inputs — the V-H5 scheduling
        trigger; assembly-checked like a ``FieldReference``
        (default: ()).
    writes : tuple[str, ...] | None
        The AUXILIARY field names this stage rewrites; ``None`` leaves
        it undeclared. Consumed by the ``time_dependent`` field lint
        (TDF-D3): a ``time_dependent``-marked field must be written by a
        SELF_UPDATE stage of its owner, resolved through ``writes=`` when
        declared (else the lint falls back to "owner has at least one
        SELF_UPDATE stage") (default: None).

    Raises
    ------
    TypeError
        If `kind` is not a ``StageKind`` member or `fn` is neither
        callable nor a method-name string.
    ValueError
        If `advances` is set on a kind that is neither ADVANCE nor
        CONSTRAINT, or `reads` is set on a non-SELF_UPDATE kind.
    """

    kind: StageKind
    fn: Callable | str
    name: str | None = None
    order: int = 0
    advances: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] | None = None
    # cadence: SELF_UPDATE only — RESERVED (CS-1), not built.

    def __post_init__(self) -> None:
        """Normalize name tuples and check local record validity."""
        if not isinstance(self.kind, StageKind):
            raise TypeError(
                f"kind must be a StageKind member, got {self.kind!r}")
        if not callable(self.fn) and not isinstance(self.fn, str):
            raise TypeError(
                f"fn must be callable or a method name, got "
                f"{self.fn!r}")
        object.__setattr__(self, "advances", tuple(self.advances))
        object.__setattr__(self, "reads", tuple(self.reads))
        if self.writes is not None:
            object.__setattr__(self, "writes", tuple(self.writes))
        if self.advances and self.kind not in (
                StageKind.ADVANCE, StageKind.CONSTRAINT):
            raise ValueError(
                f"advances= is ADVANCE/CONSTRAINT-only, got it on "
                f"{self.kind.name}")
        if self.reads and self.kind is not StageKind.SELF_UPDATE:
            raise ValueError(
                f"reads= is SELF_UPDATE-only, got it on "
                f"{self.kind.name}")


# ================================================================
#  The fr.self_update decorator
# ================================================================
def self_update(
    fn: Callable | None = None,
    *,
    reads: Iterable[str] = (),
    writes: Iterable[str] | None = None,
    cadence: object = _CADENCE_RESERVED,
) -> Callable:
    """
    Wrap a module ``self_update`` method into a SELF_UPDATE Stage.

    Description
    -----------
    Supports the bare form (``@fr.self_update``) and the
    parenthesized form (``@fr.self_update(reads=("eta",))``); the
    bare *undecorated* method is equivalent to ``reads=()``. The
    method is returned unchanged (plainly callable); the Stage
    declaration is recorded under ``STAGE_ATTRIBUTE`` and collected
    at assembly. ``reads=`` names state inputs — the V-H5 scheduling
    trigger: a module's self-update is scheduled iff one of its
    inputs is time-dependent or ``reads=`` names state components
    (assembly-checked like a ``FieldReference``).

    ``self_update`` runs per SUBSTAGE at substage time (S1, first in
    every substage — SELF_UPDATE-first is load-bearing for Tier-2
    transform determinism), stage-consistent with ``eval_params``;
    its write gate is the module's own AUXILIARY fields, applied via
    ``replace``.

    Warnings
    --------
    THE ACCUMULATION HAZARD (normative): never use ``self_update``
    for step-frequency accumulation (time means, window-mean fluxes,
    budgets). It runs per *substage* and multi-counts under
    multi-stage steppers — RK3 takes three unweighted stage-time
    samples per step, which is correct under AB3 only by accident.
    Step-frequency accumulation belongs in an S6 DIAGNOSTIC-kind
    stage — the sanctioned accumulation idiom (02_rules): a
    DIAGNOSTIC stage may read its own component's previous value and
    ``replace`` with the updated sum — step-cadence, post-NaN-seam,
    carry-resident (restart-exact). No ``fr.modules.WindowAccumulator``
    preset is provided; the idiom stands on its own. ``cadence=`` is
    RESERVED on this declaration (CS-1) and not built: passing it raises
    ``NotImplementedError``.

    Parameters
    ----------
    fn : Callable | None
        The decorated method in the bare form; ``None`` in the
        parenthesized form (default: None).
    reads : Iterable[str]
        State inputs of the update — the V-H5 scheduling trigger
        (default: ()).
    writes : Iterable[str] | None
        AUXILIARY fields this update rewrites — the ``time_dependent``
        field lint (TDF-D3); ``None`` leaves it undeclared (default:
        None).
    cadence : object
        RESERVED (CS-1), not built — any value raises.

    Returns
    -------
    Callable
        The stamped method (bare form), or the stamping decorator
        (parenthesized form).

    Raises
    ------
    NotImplementedError
        If ``cadence=`` is passed (reserved, CS-1) — use an S6
        DIAGNOSTIC-kind stage for step-frequency work instead.
    """
    if cadence is not _CADENCE_RESERVED:
        raise NotImplementedError(
            "cadence= is reserved on self_update (CS-1) and not "
            "built: self_update runs per substage; step-frequency "
            "accumulation belongs in an S6 DIAGNOSTIC-kind stage "
            "(the accumulation idiom)")

    def stamp(func: Callable) -> Callable:
        declaration = Stage(
            kind=StageKind.SELF_UPDATE,
            fn=func,
            name=func.__name__,
            reads=tuple(reads),
            writes=None if writes is None else tuple(writes),
        )
        setattr(func, STAGE_ATTRIBUTE, declaration)
        return func

    if fn is not None:
        return stamp(fn)
    return stamp
