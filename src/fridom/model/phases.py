"""
The phase axis of the stage schedule (the staggered step).

Description
-----------
``Phases`` — the declared partition of the PROGNOSTIC fields into
ordered variable groups — and ``PhaseView``, the static per-phase
view every in-trace hook reads off ``ctx.phase``. Design source:
``design/research/staggered_step_planning.md`` section 1 (the
architecture lens); the semantics amend
``design/specs/model/03_time_stepping.md`` sections 5.2/5.4 (the
canonical substage chain runs once per group inside the multistep
steppers — MITgcm's staggered baroclinic order).

``Phases`` is a frozen, hashable host record consumed at assembly and
resolved against the field table into ``Schedule.phases``
(``tuple[frozenset[str], ...]``); it never reaches jit. A single
resolved group IS the unphased path: the steppers then execute their
literal one-pass body and ``ctx.phase`` stays ``None``, so every
existing model is bitwise unchanged.
"""
# Phase axis: Phases (declaration), PhaseView (the ctx surface)
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from fridom.model.errors import AssemblyError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

# the two role-derived rules; ``None`` marks explicit groups
_STAGGERED: Final[str] = "staggered"
_TOTAL: Final[str] = "total"


# ================================================================
#  PhaseView — the ctx.phase surface
# ================================================================
class PhaseView:

    """
    The static view of the phase a hook is running in.

    Description
    -----------
    Handed to every in-trace hook through ``StepContext.phase``
    (``None`` on the unphased path). A frozen, value-hashable host
    record: it rides the ``StepContext`` pytree's STATIC aux, so a
    ``per_phase`` term reading ``ctx.phase.fields`` branches at trace
    time, never on a traced value.

    Parameters
    ----------
    index : int
        The phase's position in ``Schedule.phases`` (0-based).
    fields : Iterable[str]
        The PROGNOSTIC field names owned by this phase.
    """

    __slots__ = ("_fields", "_index")

    def __init__(self, index: int, fields: Iterable[str]) -> None:
        """Freeze the index/field-set pair; see the class docstring."""
        self._index = int(index)
        self._fields = frozenset(fields)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def index(self) -> int:
        """The phase's 0-based position in ``Schedule.phases``."""
        return self._index

    @property
    def fields(self) -> frozenset[str]:
        """The PROGNOSTIC field names this phase advances."""
        return self._fields

    # ================================================================
    #  The masking helper
    # ================================================================
    def restrict(self, names: Iterable[str]) -> tuple[str, ...]:
        """
        Keep the names this phase owns, in the given order.

        Description
        -----------
        The one-liner a ``per_phase`` term uses to skip the work of
        the other groups::

            names = (self._advected if ctx.phase is None
                     else ctx.phase.restrict(self._advected))

        The ``ctx.phase is None`` arm is load-bearing: on the
        unphased path the iteration must be the LITERAL declared
        tuple, so the trace is bitwise what it was before the phase
        axis existed.

        Parameters
        ----------
        names : Iterable[str]
            Candidate field names, in the caller's own order.

        Returns
        -------
        tuple[str, ...]
            The subset owned by this phase, order preserved.
        """
        return tuple(name for name in names if name in self._fields)

    # ================================================================
    #  Value semantics
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Equality by (index, fields) — the aux-data comparison."""
        if not isinstance(other, PhaseView):
            return NotImplemented
        return (self._index == other._index
                and self._fields == other._fields)

    def __hash__(self) -> int:
        """Value hash matching ``__eq__`` (a static treedef entry)."""
        return hash((self._index, self._fields))

    def __repr__(self) -> str:
        """Compact host-side summary."""
        names = ", ".join(sorted(self._fields))
        return f"PhaseView({self._index}, {{{names}}})"


# ================================================================
#  The hook-side helper
# ================================================================
def fields_in_phase(
    names: tuple[str, ...], ctx: object,
) -> tuple[str, ...]:
    """
    Restrict a declared field tuple to the phase a hook is in.

    Description
    -----------
    The one call a ``per_phase`` term makes to skip the other
    groups' work::

        for name in fr.model.phases.fields_in_phase(self._fields, ctx):
            ...

    On the UNPHASED path (``ctx.phase is None``) the declared tuple
    is returned UNCHANGED — the same object, so the iteration, and
    therefore the trace, is literally what it was before the phase
    axis existed. That identity is the bitwise guarantee; do not
    "simplify" it into an unconditional comprehension.

    A context object that carries no ``phase`` attribute at all is
    read as unphased: hooks are called directly with hand-rolled
    duck-typed contexts (a bare object with ``params``) in tests and
    notebooks, and "no phase axis" is the honest answer there.

    Parameters
    ----------
    names : tuple[str, ...]
        The hook's declared field names, in its own order.
    ctx : object
        The ``StepContext`` (only ``ctx.phase`` is read).

    Returns
    -------
    tuple[str, ...]
        ``names`` unchanged when unphased, else this phase's subset
        in the same order.
    """
    phase = getattr(ctx, "phase", None)
    if phase is None:
        return names
    return phase.restrict(names)


# ================================================================
#  Phases — the declaration
# ================================================================
class Phases:

    """
    The declared PROGNOSTIC partition (``fr.model.Phases``).

    Description
    -----------
    Passed to the model (``Model(..., phases=fr.model.Phases.staggered())``)
    and resolved once at assembly against the field table. Three
    spellings:

    - ``Phases.staggered()`` — role-derived, the MITgcm order: group 0
      is every PROGNOSTIC carrying the ``Velocity`` role PLUS every
      PROGNOSTIC claimed by an ADVANCE/CONSTRAINT stage's ``advances=``
      (that picks up ``ps``, ``U``, ``V`` without naming them), group 1
      is the rest (the tracers);
    - ``Phases(("u", "v", "ps"), ("b",))`` — the explicit escape
      hatch, one iterable per group, validated against the table
      (every PROGNOSTIC exactly once, no empty group);
    - ``Phases.total()`` — the explicit one-group spelling, i.e. the
      unphased schedule. It resolves to a single group, which IS the
      unphased path: the steppers run their literal one-pass body,
      ``ctx.phase`` is ``None``, and the schedule token is the
      unphased one, so ``phases=Phases.total()`` is bitwise
      ``phases=None``.

    A resolved partition of length > 1 turns the multistep steppers'
    step into a loop over the groups: per phase the canonical chain
    ``S1 SELF_UPDATE -> S1' DIAGNOSE -> S2 terms -> S3 advance ->
    S3' ADVANCE -> S4 CONSTRAINT`` at the PRE-TICK time, then one
    clock tick and one merged full-width ring level after the last
    phase.

    Frozen and value-hashable: it joins the schedule's static token
    (and the restart fingerprint) only when it resolves to more than
    one group.

    Parameters
    ----------
    *groups : Iterable[str]
        One iterable of PROGNOSTIC field names per group, in phase
        order. Empty (no groups at all) is a ``ValueError`` — the
        one-group spelling is ``Phases.total()``.

    Raises
    ------
    TypeError
        If a group is not an iterable of strings.
    ValueError
        If no group is given.
    """

    __slots__ = ("_groups", "_rule")

    def __init__(self, *groups: Iterable[str]) -> None:
        """Record the explicit groups; see the class docstring."""
        if not groups:
            raise ValueError(
                "Phases() needs at least one group of PROGNOSTIC "
                "names, e.g. Phases(('u', 'v', 'ps'), ('b',)); the "
                "one-group (unphased) spelling is Phases.total()")
        resolved: list[tuple[str, ...]] = []
        for index, group in enumerate(groups):
            if isinstance(group, str):
                raise TypeError(
                    f"Phases group {index} is the bare string "
                    f"{group!r}; a group is an ITERABLE of names — "
                    f"write ({group!r},), never {group!r}")
            names = tuple(group)
            if any(not isinstance(name, str) for name in names):
                raise TypeError(
                    f"Phases group {index} must contain field-name "
                    f"strings, got {names!r}")
            resolved.append(names)
        self._groups: tuple[tuple[str, ...], ...] | None = tuple(
            resolved)
        self._rule: str | None = None

    # ================================================================
    #  The derived spellings
    # ================================================================
    @classmethod
    def staggered(cls) -> Phases:
        """
        Return the role-derived two-group (MITgcm) partition.

        Description
        -----------
        Group 0 = the ``Velocity``-role PROGNOSTIC fields plus every
        PROGNOSTIC claimed by an ADVANCE or CONSTRAINT stage's
        ``advances=`` (the free-surface ``ps``, the split-explicit
        transports ``U``/``V``); group 1 = the remaining PROGNOSTIC
        fields (the tracers). Resolved at assembly, so adding a
        tracer module needs no edit here.

        Returns
        -------
        Phases
            The role-derived declaration.
        """
        return cls._derived(_STAGGERED)

    @classmethod
    def total(cls) -> Phases:
        """
        Return the one-group partition — the unphased schedule.

        Description
        -----------
        The explicit spelling of "no phase split": it resolves to a
        single group holding every PROGNOSTIC field, which the
        steppers execute as the literal unphased body (``ctx.phase``
        stays ``None``). Bitwise identical to ``phases=None``, and it
        keys the same jit-cache entry.

        Returns
        -------
        Phases
            The one-group declaration.
        """
        return cls._derived(_TOTAL)

    @classmethod
    def _derived(cls, rule: str) -> Phases:
        """Build a rule-derived declaration (no explicit groups)."""
        obj = object.__new__(cls)
        # __init__ takes the explicit groups; a derived spelling
        # bypasses it, so the two slots are set here directly
        object.__setattr__(obj, "_groups", None)
        object.__setattr__(obj, "_rule", rule)
        return obj

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def groups(self) -> tuple[tuple[str, ...], ...] | None:
        """The explicitly declared groups; ``None`` when derived."""
        return self._groups

    @property
    def rule(self) -> str | None:
        """The derivation rule (``"staggered"``/``"total"``) or None."""
        return self._rule

    # ================================================================
    #  Resolution (assembly step 6b)
    # ================================================================
    def resolve(
        self,
        prognostic: tuple[str, ...],
        *,
        velocity: Iterable[str] = (),
        claimed: Iterable[str] = (),
    ) -> tuple[frozenset[str], ...]:
        """
        Resolve the declaration against the field table.

        Description
        -----------
        Returns the ordered groups as frozensets of PROGNOSTIC names.
        Every PROGNOSTIC field belongs to exactly one group;
        AUXILIARY and DIAGNOSTIC fields belong to none (stages
        writing them run per their kind rule). A resolved length of 1
        is the unphased path.

        Parameters
        ----------
        prognostic : tuple[str, ...]
            The PROGNOSTIC field names, declaration order.
        velocity : Iterable[str], optional
            The PROGNOSTIC names carrying the ``Velocity`` role — the
            seed of ``staggered()``'s group 0 (default: ()).
        claimed : Iterable[str], optional
            The PROGNOSTIC names claimed by an ADVANCE/CONSTRAINT
            stage's ``advances=`` — also group 0 under
            ``staggered()`` (default: ()).

        Returns
        -------
        tuple[frozenset[str], ...]
            The resolved partition, in phase order.

        Raises
        ------
        AssemblyError
            On an unknown or non-PROGNOSTIC name, a field named twice
            or not at all, or an empty group in a multi-group
            partition.
        """
        names = tuple(prognostic)
        if self._rule == _TOTAL:
            return (frozenset(names),)
        if self._rule == _STAGGERED:
            groups = self._staggered_groups(names, velocity, claimed)
        else:
            groups = self._explicit_groups(names)
        self._check_partition(groups, names)
        return groups

    def _staggered_groups(
        self,
        prognostic: tuple[str, ...],
        velocity: Iterable[str],
        claimed: Iterable[str],
    ) -> tuple[frozenset[str], ...]:
        """Derive the momentum/tracer split from roles and claims."""
        known = set(prognostic)
        first = {name for name in (*velocity, *claimed)
                 if name in known}
        rest = known - first
        if not first or not rest:
            raise AssemblyError(
                "Phases.staggered() derives a single non-empty group "
                f"from this assembly (momentum/barotropic: "
                f"{tuple(sorted(first))}, tracers: "
                f"{tuple(sorted(rest))}): the staggered step needs a "
                "tracer group to stagger against the momentum one. "
                "Drop phases= (or pass fr.model.Phases.total()) for "
                "the unphased schedule, or name the groups "
                "explicitly with fr.model.Phases((...), (...))")
        return (frozenset(first), frozenset(rest))

    def _explicit_groups(
        self, prognostic: tuple[str, ...],
    ) -> tuple[frozenset[str], ...]:
        """Validate the declared groups against the PROGNOSTIC set."""
        known = set(prognostic)
        groups: list[frozenset[str]] = []
        for index, group in enumerate(self._groups or ()):
            stray = tuple(name for name in group if name not in known)
            if stray:
                raise AssemblyError(
                    f"Phases group {index} names {stray}, which are "
                    f"not declared PROGNOSTIC fields (PROGNOSTIC: "
                    f"{prognostic}); phases partition the PROGNOSTIC "
                    "fields only — AUXILIARY and DIAGNOSTIC fields "
                    "belong to no group")
            if len(set(group)) != len(group):
                raise AssemblyError(
                    f"Phases group {index} repeats a field name "
                    f"({group!r}); every PROGNOSTIC field belongs to "
                    "exactly one group")
            groups.append(frozenset(group))
        return tuple(groups)

    @staticmethod
    def _check_partition(
        groups: tuple[frozenset[str], ...],
        prognostic: tuple[str, ...],
    ) -> None:
        """Require an exact, non-degenerate partition (multi-group)."""
        if len(groups) > 1:
            empty = tuple(index for index, group in enumerate(groups)
                          if not group)
            if empty:
                raise AssemblyError(
                    f"Phases group(s) {empty} are empty; an empty "
                    "phase runs the whole substage chain and "
                    "advances nothing — drop the group (or pass "
                    "fr.model.Phases.total() for the unphased "
                    "schedule)")
        seen: set[str] = set()
        for group in groups:
            twice = tuple(sorted(seen & group))
            if twice:
                raise AssemblyError(
                    f"PROGNOSTIC field(s) {twice} appear in more "
                    "than one phase group; every PROGNOSTIC field "
                    "belongs to exactly one group")
            seen |= group
        missing = tuple(name for name in prognostic if name not in seen)
        if missing:
            raise AssemblyError(
                f"PROGNOSTIC field(s) {missing} belong to no phase "
                "group; the groups must partition the PROGNOSTIC "
                f"fields exactly (PROGNOSTIC: {prognostic})")

    # ================================================================
    #  Value semantics (the declaration joins no hashed record until
    #  it is resolved, but it must still compare by value)
    # ================================================================
    def _token(self) -> tuple:
        """Return the hashable structural identity."""
        return (self._rule, self._groups)

    def __eq__(self, other: object) -> bool:
        """Structural equality (rule and declared groups)."""
        if not isinstance(other, Phases):
            return NotImplemented
        return self._token() == other._token()

    def __hash__(self) -> int:
        """Structural hash matching ``__eq__``."""
        return hash(self._token())

    def __repr__(self) -> str:
        """Round-tripping repr of the declaration."""
        if self._rule is not None:
            return f"Phases.{self._rule}()"
        groups = ", ".join(repr(group) for group in self._groups or ())
        return f"Phases({groups})"
