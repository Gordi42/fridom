"""
Composable term predicates (``fr.terms``) — task 2.8.

Description
-----------
The five predicate leaves of ``design/specs/model/08_state_transforms.md``
§10.4 — ``linear``, ``explicit``/``implicit``, ``owned_by(Type)``,
``named(*keys)``, ``advancing(*fields)`` — combined with ``& | ~`` into
frozen, reprable expression trees with a stable ``fingerprint_token``.
They select tendency **terms** (never declarations/stages): a variant
filters at assembly (``model.variant(term_filter=...)``), and
``model.tendency(filter=...)`` restricts the evaluated terms.

A predicate is a callable ``(key, term, module=None) -> bool`` where
``key`` is the ``"Module/term"`` attribution key, ``term`` the
``TendencyTerm``, and ``module`` its owning module (used by
``owned_by`` for the ``isinstance`` check). ``wants_module = True``
signals the composer/host to pass the third argument; the composer's
legacy two-argument callables keep working. ``fr.linearize(model)``
is ``model.variant(term_filter=fr.terms.linear)``.

Owning class spec: ``design/specs/model/classes/declarations.md``
(``fr.terms`` = ``fr.model.term_predicates``).
"""
# Wave 7 A: TermPredicate + linear/explicit/implicit/owned_by/named/
#    advancing, combined with & | ~
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from fridom.framework2.model.terms import Treatment

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.model.terms import TendencyTerm


# ================================================================
#  The match context (one argument -> no unused-argument churn)
# ================================================================
class _TermContext(NamedTuple):

    """The evaluated term plus its attribution key and owning module."""

    key: str
    term: TendencyTerm
    module: object


# ================================================================
#  TermPredicate — the base of the frozen expression tree
# ================================================================
class TermPredicate(ABC):

    """
    A composable term predicate (``(key, term, module) -> bool``).

    Description
    -----------
    Frozen, reprable, and combinable with ``&`` / ``|`` / ``~``. Each
    node exposes a stable :meth:`fingerprint_token` (the variant
    provenance token that enters the assembly fingerprint) and
    :meth:`referenced_names` (the ``named`` keys in the tree, for the
    unknown-key build check). ``wants_module = True`` requests the
    owning module as the third call argument.
    """

    wants_module: ClassVar[bool] = True

    @abstractmethod
    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether ``ctx`` satisfies this leaf/node."""

    @abstractmethod
    def fingerprint_token(self) -> str:
        """Return the stable provenance token (enters the fingerprint)."""

    def referenced_names(self) -> frozenset[str]:
        """Return the ``named`` keys in this tree (default: empty)."""
        return frozenset()

    def __call__(
        self, key: str, term: TendencyTerm, module: object = None,
    ) -> bool:
        """Evaluate the predicate on one attributed term."""
        return self._matches(_TermContext(key, term, module))

    # -- the canonical filter token (assembly reads ``.token``) ------
    @property
    def token(self) -> str:
        """Return the canonical filter token (the fingerprint)."""
        return self.fingerprint_token()

    # ================================================================
    #  Combinators (frozen trees)
    # ================================================================
    def __and__(self, other: object) -> TermPredicate:
        """Build the conjunction ``self & other``."""
        if not isinstance(other, TermPredicate):
            return NotImplemented
        return And(self, other)

    def __or__(self, other: object) -> TermPredicate:
        """Build the disjunction ``self | other``."""
        if not isinstance(other, TermPredicate):
            return NotImplemented
        return Or(self, other)

    def __invert__(self) -> TermPredicate:
        """Build the negation ``~self``."""
        return Not(self)

    def __repr__(self) -> str:
        """Render as ``fr.terms.<fingerprint token>``."""
        return f"fr.terms.{self.fingerprint_token()}"


# ================================================================
#  Combinator nodes
# ================================================================
class And(TermPredicate):

    """Conjunction of two predicates."""

    def __init__(
        self, left: TermPredicate, right: TermPredicate,
    ) -> None:
        """Store the two operands."""
        self._left = left
        self._right = right

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether both operands match."""
        return (self._left(*ctx) and self._right(*ctx))

    def fingerprint_token(self) -> str:
        """Return ``(left & right)``."""
        return (f"({self._left.fingerprint_token()} & "
                f"{self._right.fingerprint_token()})")

    def referenced_names(self) -> frozenset[str]:
        """Return the union of the operands' referenced names."""
        return (self._left.referenced_names()
                | self._right.referenced_names())


class Or(TermPredicate):

    """Disjunction of two predicates."""

    def __init__(
        self, left: TermPredicate, right: TermPredicate,
    ) -> None:
        """Store the two operands."""
        self._left = left
        self._right = right

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether either operand matches."""
        return (self._left(*ctx) or self._right(*ctx))

    def fingerprint_token(self) -> str:
        """Return ``(left | right)``."""
        return (f"({self._left.fingerprint_token()} | "
                f"{self._right.fingerprint_token()})")

    def referenced_names(self) -> frozenset[str]:
        """Return the union of the operands' referenced names."""
        return (self._left.referenced_names()
                | self._right.referenced_names())


class Not(TermPredicate):

    """Negation of a predicate."""

    def __init__(self, inner: TermPredicate) -> None:
        """Store the negated operand."""
        self._inner = inner

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the operand does not match."""
        return not self._inner(*ctx)

    def fingerprint_token(self) -> str:
        """Return ``~inner``."""
        return f"~{self._inner.fingerprint_token()}"

    def referenced_names(self) -> frozenset[str]:
        """Return the operand's referenced names."""
        return self._inner.referenced_names()


# ================================================================
#  Leaves
# ================================================================
class _Linear(TermPredicate):

    """Terms tagged ``linear=True`` (strict linear-in-state)."""

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the term declares ``linear=True``."""
        return bool(getattr(ctx.term, "linear", False))

    def fingerprint_token(self) -> str:
        """Return ``linear``."""
        return "linear"


class _Explicit(TermPredicate):

    """Terms with ``EXPLICIT`` treatment."""

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the term is ``EXPLICIT``."""
        return ctx.term.treatment is Treatment.EXPLICIT

    def fingerprint_token(self) -> str:
        """Return ``explicit``."""
        return "explicit"


class _Implicit(TermPredicate):

    """Terms with ``IMPLICIT`` treatment."""

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the term is ``IMPLICIT``."""
        return ctx.term.treatment is Treatment.IMPLICIT

    def fingerprint_token(self) -> str:
        """Return ``implicit``."""
        return "implicit"


class OwnedBy(TermPredicate):

    """Terms owned by a module of the given type (``isinstance``)."""

    def __init__(self, module_type: type) -> None:
        """Store the module type to match by ``isinstance``."""
        if not isinstance(module_type, type):
            raise TypeError(
                f"owned_by takes a module type, got {module_type!r}")
        self._type = module_type

    def _matches(self, ctx: _TermContext) -> bool:
        """Match by ``isinstance`` (module known) or by class name."""
        if ctx.module is not None:
            return isinstance(ctx.module, self._type)
        owner = ctx.key.split("/", 1)[0]
        return owner in (self._type.__name__, self._type.__qualname__)

    def fingerprint_token(self) -> str:
        """Return ``owned_by(qualname)`` (qualified class name)."""
        return f"owned_by({self._type.__qualname__})"


class Named(TermPredicate):

    """Terms whose attribution key is one of the given keys."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        """Store the ``"Module/term"`` keys to match."""
        if not keys:
            raise ValueError(
                'named() needs at least one "Module/term" key')
        self._keys = tuple(keys)

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the attribution key is one of the keys."""
        return ctx.key in self._keys

    def referenced_names(self) -> frozenset[str]:
        """Return the named keys (for the unknown-key build check)."""
        return frozenset(self._keys)

    def fingerprint_token(self) -> str:
        """Return ``named(sorted keys)``."""
        return f"named({','.join(sorted(self._keys))})"


class Advancing(TermPredicate):

    """Terms that advance any of the given PROGNOSTIC fields."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        """Store the advanced-field names to match."""
        if not fields:
            raise ValueError(
                "advancing() needs at least one field name")
        self._fields = frozenset(fields)

    def _matches(self, ctx: _TermContext) -> bool:
        """Return whether the term's advances intersect the set."""
        advances = getattr(ctx.term, "advances", None) or ()
        return bool(self._fields & set(advances))

    def fingerprint_token(self) -> str:
        """Return ``advancing(sorted fields)``."""
        return f"advancing({','.join(sorted(self._fields))})"


# ================================================================
#  The public surface (fr.terms.*)
# ================================================================
#: match terms tagged ``linear=True``
linear: _Linear = _Linear()
#: match ``EXPLICIT``-treatment terms
explicit: _Explicit = _Explicit()
#: match ``IMPLICIT``-treatment terms
implicit: _Implicit = _Implicit()


def owned_by(module_type: type) -> OwnedBy:
    """
    Match terms owned by a module of ``module_type`` (``isinstance``).

    Parameters
    ----------
    module_type : type
        The module class (subclasses match).

    Returns
    -------
    OwnedBy
        The predicate leaf.
    """
    return OwnedBy(module_type)


def named(*keys: str) -> Named:
    """
    Match terms by their ``"Module/term"`` attribution key.

    Parameters
    ----------
    *keys : str
        One or more attribution keys; unknown keys are a build-time
        error (checked where the full key set is known).

    Returns
    -------
    Named
        The predicate leaf.
    """
    return Named(keys)


def advancing(*fields: str) -> Advancing:
    """
    Match terms that advance any of the named PROGNOSTIC fields.

    Description
    -----------
    Splits one module's terms by advanced subset (e.g. a Smagorinsky
    stress term vs a kappa mixing term).

    Parameters
    ----------
    *fields : str
        One or more advanced PROGNOSTIC field names.

    Returns
    -------
    Advancing
        The predicate leaf.
    """
    return Advancing(fields)


def linearize(model: object, *, name: str | None = None) -> object:
    """
    Build the linear variant ``model.variant(term_filter=linear)``.

    Description
    -----------
    Top-level sugar (08 §10.4): keeps exactly the ``linear=True``
    terms — a full lifecycle citizen ``fr.Model``, not a transform.
    Consumes the declared ``linear`` tag (V-S3: a scheme's linear
    background-advection piece must be a separate ``linear=True`` term
    or ``linearize`` drops it).

    Parameters
    ----------
    model : Model
        The parent model to linearize.
    name : str | None, optional
        The variant name (default: ``"{parent}/linear"``).

    Returns
    -------
    Model
        The linear variant.
    """
    if name is None:
        parent = getattr(model, "name", None)
        name = f"{parent}/linear" if parent else "linear"
    return model.variant(term_filter=linear, name=name)
