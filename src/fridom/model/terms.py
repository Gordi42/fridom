"""
Tendency-term declarations.

Description
-----------
``Treatment``, ``TendencyTerm``, and the ``@fr.model.term`` decorator: a
module's tendency contribution as declared frozen data — the
FieldDeclaration pattern applied to behavior. Owning class spec:
``design/specs/model/classes/declarations.md``; design source
``design/specs/model/03_time_stepping.md`` section 5.1.

Terms are transient assembly inputs: plain frozen host objects, never
pytrees, never in the carry, never reaching jit. The composed step
closes over module slot indices and the unbound ``fn``; all
attribution, validation, and accumulation-order machinery lives in
the ``TendencyComposer`` (wave 3+) — this vocabulary only records.

THE PHASE AXIS (``fr.model.Phases``, ``phases.py``). Under a
multi-group partition the substage chain runs once per PROGNOSTIC
group, so every term must belong to a group: a term's dry-run write
set, restricted to PROGNOSTIC keys, has to lie inside ONE group —
otherwise its contribution would be summed into a group it is not
being integrated with. A term that genuinely writes across groups
(``Advection`` writes ``{u, v, b}`` in one hook,
``MeshVelocityCorrection`` corrects every prognostic,
``ThermalWindBackground`` writes ``{u, b}``) declares
``per_phase=True``: it is then EVALUATED ONCE PER PHASE it touches
and only that phase's keys are kept — the other keys are DISCARDED,
never zeroed, so they never enter any sum. ``ctx.phase``
(a ``PhaseView``, ``None`` on the unphased path) lets such a term
skip the other groups' work instead of computing and dropping it::

    names = (self._advected if ctx.phase is None
             else ctx.phase.restrict(self._advected))

A straddling term WITHOUT ``per_phase=True`` is a taught assembly
error. On the unphased path ``per_phase`` changes nothing: the term
runs once, ``ctx.phase`` is ``None``, and the trace is unchanged.
"""
# Wave 2 C: Treatment, TendencyTerm, term
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

from fridom.model.implicit import ImplicitOperator

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable


# ================================================================
#  Treatment
# ================================================================
class Treatment(Enum):

    """
    Integration treatment of a tendency term (closed, iteration 1).

    Description
    -----------
    Treatment is author-declared on the term; the user override lives
    on the module constructor
    (``VerticalMixing(kv=..., treatment=fr.model.IMPLICIT)``), never on the
    Model. An IMPLICIT term under a purely explicit stepper is an
    assembly error, never a silent demotion.
    """

    EXPLICIT = auto()
    IMPLICIT = auto()


# module-level aliases, re-exported as fr.model.EXPLICIT / fr.model.IMPLICIT
EXPLICIT: Final[Treatment] = Treatment.EXPLICIT
IMPLICIT: Final[Treatment] = Treatment.IMPLICIT


# attribute under which @fr.model.term stamps the declaration onto the
# method; the wave-3 collector (Module.tendency_terms) scans for it
# in definition order
TERM_ATTRIBUTE: Final[str] = "__fridom_term__"


# ================================================================
#  TendencyTerm
# ================================================================
@dataclass(frozen=True)
class TendencyTerm:

    """
    One declared tendency contribution (transient assembly input).

    Description
    -----------
    A frozen host object consumed by assembly and discarded — never a
    pytree, never in the carry. ``fn`` is stored UNBOUND and paired
    with a module *slot* at compose time: the composer calls
    ``term.fn(carry.modules[slot], state, ctx)`` inside the trace (a
    bound method would capture the assembly-time module instance
    while live parameters ride the carry — the D2 aliasing trap).
    Bound-ness is validated at assembly (wave 3+); the vocabulary
    just records.

    The attribution key is ``"Module/term"``, composed by the
    collector from the owning module's class name and ``name``.
    Contribution dicts key PROGNOSTIC components only and are applied
    via ``VectorField.add`` in deterministic order — terms only ever
    add; anything that overwrites is a stage.

    Parameters
    ----------
    name : str
        The term part of the ``"Module/term"`` attribution key.
    fn : Callable | None
        UNBOUND hook ``(module, state, ctx) -> dict`` mapping
        PROGNOSTIC names to increments; ``None`` for implicit terms
        (write-once: the explicit path is then derived from
        ``implicit.apply``, so flipping the treatment cannot
        desynchronize the two). Default: ``None``.
    treatment : Treatment
        Integration treatment; ``IMPLICIT`` without ``implicit=`` is
        an assembly error (checked at assembly, not here). Default:
        ``Treatment.EXPLICIT``.
    advances : tuple[str, ...] | None
        Optional-declared PROGNOSTIC subset this term advances;
        always dry-run-verified at assembly (a mismatch is an
        assembly error). Default: ``None`` (derived by the dry run).
    transports : tuple[str, ...]
        Declared-only intent feeding the D1.4 coverage lint (not
        derivable: diffusion and advection both write ``b``; only
        one transports it). Default: ``()``.
    implicit : ImplicitOperator | None
        The implicit-operator slot; required iff
        ``treatment == IMPLICIT`` (assembly-checked). Default:
        ``None``.
    linear : bool
        Strict tag: linear in the state at fixed parameters/aux;
        state-independent forcing is *not* linear. Consumers:
        ``fr.model.term_predicates.linear``, ``fr.model.linearize``,
        the IMEX partition sanity. Default: ``False``.
    linear_params : tuple[str, ...]
        Declared parameter names the term's *linear operator* depends
        on (TDF-D4): the honesty seam for a frozen-``L`` (exponential)
        stepper. ``Module.time_dependent_linear_parameters`` resolves
        each to the module's own leaf and reports it when the leaf is a
        ``TimeDependent`` curve. Annotating a name that resolves to a
        plain constant is free (the guard only fires on actual time
        dependence), so a term lists every *potentially* time-dependent
        parameter dependency. Default: ``()``.
    linear_fields : tuple[str, ...]
        Declared field names the term's linear operator depends on
        (TDF-D4): resolved to the owning module's ``FieldDeclaration``
        and reported when it carries the ``time_dependent`` marker
        (a field-carried time dependence in ``L``). Default: ``()``.
    per_phase : bool
        Whether this term may write ACROSS the phase groups (module
        docstring). ``False`` — the default — requires the write set
        to lie inside one group (a straddle is an assembly error);
        ``True`` evaluates the term once per phase it intersects and
        keeps only that phase's keys. No effect on the unphased
        path. Default: ``False``.

    Raises
    ------
    TypeError
        If ``name`` is not a non-empty string, ``treatment`` is not
        a ``Treatment`` member, ``fn`` is not callable, or
        ``implicit`` does not satisfy the ``ImplicitOperator``
        protocol.
    ValueError
        If both ``fn`` and ``implicit`` are ``None`` (a term with no
        behavior records nothing).
    """

    name: str
    fn: Callable | None = None
    treatment: Treatment = Treatment.EXPLICIT
    advances: tuple[str, ...] | None = None
    transports: tuple[str, ...] = ()
    implicit: ImplicitOperator | None = None
    linear: bool = False
    linear_params: tuple[str, ...] = ()
    linear_fields: tuple[str, ...] = ()
    per_phase: bool = False

    def __post_init__(self) -> None:
        """Normalize name tuples and check local record validity."""
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(
                f"term name must be a non-empty string, got "
                f"{self.name!r}")
        if not isinstance(self.treatment, Treatment):
            raise TypeError(
                f"treatment must be a Treatment member, got "
                f"{self.treatment!r}")
        if self.fn is None and self.implicit is None:
            raise ValueError(
                f"term {self.name!r} declares no behavior: provide "
                "fn=, or implicit= (from which the explicit path is "
                "derived)")
        if self.fn is not None and not callable(self.fn):
            raise TypeError(
                f"term {self.name!r}: fn must be callable, got "
                f"{self.fn!r}")
        if (self.implicit is not None
                and not isinstance(self.implicit, ImplicitOperator)):
            raise TypeError(
                f"term {self.name!r}: implicit= must satisfy the "
                f"ImplicitOperator protocol, got {self.implicit!r}")
        if self.advances is not None:
            object.__setattr__(self, "advances", tuple(self.advances))
        object.__setattr__(self, "transports", tuple(self.transports))
        object.__setattr__(
            self, "linear_params",
            tuple(str(name) for name in self.linear_params))
        object.__setattr__(
            self, "linear_fields", tuple(self.linear_fields))
        if not isinstance(self.per_phase, bool):
            raise TypeError(
                f"term {self.name!r}: per_phase must be a bool, got "
                f"{self.per_phase!r}")

    def __repr__(self) -> str:
        """Return a compact record repr (assembly logs terms)."""
        parts = [f"name={self.name!r}",
                 f"treatment={self.treatment.name}"]
        if self.fn is not None:
            parts.append(f"fn={getattr(self.fn, '__name__', self.fn)}")
        if self.advances is not None:
            parts.append(f"advances={self.advances!r}")
        if self.transports:
            parts.append(f"transports={self.transports!r}")
        if self.implicit is not None:
            parts.append(f"implicit={type(self.implicit).__name__}")
        if self.linear:
            parts.append("linear=True")
        if self.linear_params:
            parts.append(f"linear_params={self.linear_params!r}")
        if self.linear_fields:
            parts.append(f"linear_fields={self.linear_fields!r}")
        if self.per_phase:
            parts.append("per_phase=True")
        return f"TendencyTerm({', '.join(parts)})"


# ================================================================
#  The @fr.model.term decorator
# ================================================================
def term(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    treatment: Treatment = Treatment.EXPLICIT,
    advances: Iterable[str] | None = None,
    transports: Iterable[str] = (),
    implicit: ImplicitOperator | None = None,
    linear: bool = False,
    linear_params: Iterable[str] = (),
    linear_fields: Iterable[str] = (),
    per_phase: bool = False,
) -> Callable:
    """
    Stamp a module method as a ``TendencyTerm`` (``@fr.model.term``).

    Description
    -----------
    Supports both the bare form (``@fr.model.term``) and the parenthesized
    form (``@fr.model.term(advances=("w",), linear=True)``). The term name
    defaults to the method name; the method itself is returned
    unchanged, so it stays plainly callable (the halo trace runs it
    un-jitted). The declaration is recorded on the function under
    ``TERM_ATTRIBUTE``; ``Module.tendency_terms()`` (wave 3) collects
    stamped methods in definition order.

    The stamped function is the plain class-body function — i.e. it
    is stored UNBOUND, per the aliasing rule; assembly validates that
    collected ``fn`` slots carry no ``__self__``.

    Parameters
    ----------
    fn : Callable | None
        The decorated method in the bare form; ``None`` in the
        parenthesized form (default: None).
    name : str | None
        Term part of the attribution key; defaults to the method
        name (default: None).
    treatment : Treatment
        Author-declared integration treatment (default:
        ``Treatment.EXPLICIT``).
    advances : Iterable[str] | None
        Optional-declared advanced PROGNOSTIC subset (default: None).
    transports : Iterable[str]
        Declared transport intent for the coverage lint
        (default: ()).
    implicit : ImplicitOperator | None
        Implicit-operator slot (default: None).
    linear : bool
        Strict linear-in-state tag (default: False).
    linear_params : Iterable[str]
        Parameter names the linear operator depends on, resolved by
        ``Module.time_dependent_linear_parameters`` for the frozen-``L``
        guard (TDF-D4); free to over-list (default: ()).
    linear_fields : Iterable[str]
        Field names the linear operator depends on, resolved against
        the ``time_dependent`` declaration marker (default: ()).
    per_phase : bool
        Allow the write set to straddle the phase groups: the term
        is then evaluated once per phase it touches and masked to
        that phase's keys (module docstring). No effect on the
        unphased path (default: False).

    Returns
    -------
    Callable
        The stamped method (bare form), or the stamping decorator
        (parenthesized form).
    """
    def stamp(func: Callable) -> Callable:
        declaration = TendencyTerm(
            name=name if name is not None else func.__name__,
            fn=func,
            treatment=treatment,
            advances=advances,
            transports=transports,
            implicit=implicit,
            linear=linear,
            linear_params=tuple(linear_params),
            linear_fields=tuple(linear_fields),
            per_phase=per_phase,
        )
        setattr(func, TERM_ATTRIBUTE, declaration)
        return func

    if fn is not None:
        return stamp(fn)
    return stamp
