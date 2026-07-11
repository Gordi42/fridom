"""
Tendency-term declarations.

Description
-----------
``Treatment``, ``TendencyTerm``, and the ``@fr.term`` decorator: a
module's tendency contribution as declared frozen data — the
FieldDeclaration pattern applied to behavior. Owning class spec:
``design/specs/model/classes/declarations.md``; design source
``design/specs/model/03_time_stepping.md`` section 5.1.

Terms are transient assembly inputs: plain frozen host objects, never
pytrees, never in the carry, never reaching jit. The composed step
closes over module slot indices and the unbound ``fn``; all
attribution, validation, and accumulation-order machinery lives in
the ``TendencyComposer`` (wave 3+) — this vocabulary only records.
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
    (``VerticalMixing(kv=..., treatment=fr.IMPLICIT)``), never on the
    Model. An IMPLICIT term under a purely explicit stepper is an
    assembly error, never a silent demotion.
    """

    EXPLICIT = auto()
    IMPLICIT = auto()


# module-level aliases, re-exported as fr.EXPLICIT / fr.IMPLICIT
EXPLICIT: Final[Treatment] = Treatment.EXPLICIT
IMPLICIT: Final[Treatment] = Treatment.IMPLICIT


# attribute under which @fr.term stamps the declaration onto the
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
        ``fr.terms.linear``, ``fr.linearize``, the IMEX partition
        sanity. Default: ``False``.

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
        return f"TendencyTerm({', '.join(parts)})"


# ================================================================
#  The @fr.term decorator
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
) -> Callable:
    """
    Stamp a module method as a ``TendencyTerm`` (``@fr.term``).

    Description
    -----------
    Supports both the bare form (``@fr.term``) and the parenthesized
    form (``@fr.term(advances=("w",), linear=True)``). The term name
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
        )
        setattr(func, TERM_ATTRIBUTE, declaration)
        return func

    if fn is not None:
        return stamp(fn)
    return stamp
