r"""
The realized-map category: ``RealizedMap`` and lazy compositions.

Description
-----------
Owning design: ``design/decisions/symbol_stack_design.md`` (the
realized-map algebra, grades, explicit materialization) and
``design/plans/active/composition_refactor_plan.md`` (section 4, "The
``RealizedMap`` category"; stage S1). A **realized map** is anything
that maps ``ScalarField -> ScalarField`` with a *fixed, already-
resolved* ``(domain, codomain)`` coefficient-space tag pair: the
diagonal ``Symbol``, a grid-bound transform (wrapped as a
``BoundTransform``), and the lazy ``RealizedComposite`` /
``RealizedSum`` chains they build under ``@`` / ``+``.

``RealizedMap`` is a ``runtime_checkable`` :class:`typing.Protocol`
(**not** a base class), so a ``@final`` type like ``Symbol`` conforms
by having the surface without any inheritance, ``jaxify``, or MRO
friction. Composition is well-typed through the S0a factor-wise tag
validator (:func:`compose_spaces` / :func:`union_spaces`): a
representation mismatch — a spectral ``Symbol`` against a physical
stencil with no transform between — is a tag error caught for free.

Crossing from an *unmaterialized* :class:`Operator` recipe into this
category is **explicit** (decision A): ``symbol @ recipe`` raises a
taught :class:`SpaceMismatchError` pointing at
``recipe.eigenvalues(grid, space)`` — the one labelled door.
"""
# Wave: composition refactor S1 (RealizedMap category)
from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import fridom.framework as fr
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.spaces.composition import (
    compose_spaces,
    union_spaces,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.mixed import ComposedTransform
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


# ================================================================
#  The category interface
# ================================================================
@runtime_checkable
class RealizedMap(Protocol):

    r"""
    The common surface of a grid-bound, data-carrying field map.

    Description
    -----------
    A ``runtime_checkable`` protocol (structural, not nominal): any
    object with ``domain`` / ``codomain`` fixed-tag accessors, a
    field ``__call__``, and the ``@`` / ``+`` / ``inverse`` / ``conj``
    algebra is a realized map. ``Symbol`` conforms directly (its
    ``domain`` is an alias of ``space``); ``BoundTransform``,
    ``RealizedComposite`` and ``RealizedSum`` are the other members.
    A static :class:`Operator` recipe is **not** a realized map: it
    lacks the fixed tags and the cheap inverse, and must be
    materialized (``op.eigenvalues(grid, space)``) to cross over.
    """

    @property
    def domain(self) -> SpaceLike:
        """The fixed (already-resolved) domain coefficient tag."""
        ...

    @property
    def codomain(self) -> SpaceLike:
        """The fixed (already-resolved) codomain coefficient tag."""
        ...

    def __call__(self, f: FieldLike) -> FieldLike:
        """Apply the map to a field on ``domain``."""
        ...

    def __matmul__(self, other: RealizedMap) -> RealizedMap:
        """Compose ``(A @ B)(f) == A(B(f))``."""
        ...

    def __add__(self, other: RealizedMap) -> RealizedMap:
        """Sum ``(A + B)(f) == A(f) + B(f)``."""
        ...

    def inverse(self, where_zero: complex = 0.0) -> RealizedMap:
        """Return the (pseudo-)inverse map."""
        ...

    def conj(self) -> RealizedMap:
        """Return the complex-conjugate map."""
        ...


# ================================================================
#  Cross-type dispatch helpers (shared by every realized map)
# ================================================================
_MATERIALIZE_MSG = (
    "cannot compose a realized map with an unmaterialized operator "
    "({op!r}); materialize it first: op.eigenvalues(grid, space)")


def _guard_operator(other: object, *, operation: str) -> None:
    """
    Raise the taught materialization guard if ``other`` is a recipe.

    Description
    -----------
    The explicit-materialization guard (decision A): a static
    :class:`Operator` is a grid-free recipe, not a realized map, so
    composing it with a realized map without first materializing it
    (``op.eigenvalues(grid, space)``) is a :class:`SpaceMismatchError`.
    A non-operator, non-realized operand is left alone (the dunder
    returns ``NotImplemented``).

    Parameters
    ----------
    other : object
        The prospective composition operand.
    operation : str
        The label recorded on a raised ``SpaceMismatchError``.
    """
    from fridom.spatial.operators.base import (  # noqa: PLC0415
        Operator,
    )
    if isinstance(other, Operator):
        raise SpaceMismatchError(
            _MATERIALIZE_MSG.format(op=other),
            right=other, operation=operation)


def realized_matmul(
    outer: RealizedMap, inner: object,
) -> RealizedMap:
    """
    Compose ``outer @ inner`` where ``outer`` is a realized map.

    Description
    -----------
    A realized ``inner`` builds (or eager-fuses into) a
    :func:`compose_realized` chain; an unmaterialized :class:`Operator`
    raises the materialization guard; anything else yields
    ``NotImplemented`` so Python can try the reflected dunder.

    Parameters
    ----------
    outer : RealizedMap
        The outer (applied-last) realized map.
    inner : object
        The inner (applied-first) operand.

    Returns
    -------
    RealizedMap
        The composed map (or ``NotImplemented``).
    """
    if isinstance(inner, RealizedMap):
        return compose_realized(outer, inner)
    _guard_operator(inner, operation="RealizedMap.__matmul__")
    return NotImplemented


def realized_rmatmul(
    inner: RealizedMap,  # noqa: ARG001 — signature symmetry with matmul
    outer: object,
) -> RealizedMap:
    """
    Reflected ``outer @ inner`` where ``inner`` is a realized map.

    Description
    -----------
    Reached only when ``outer`` did not implement ``@`` for
    ``inner`` — so ``outer`` is never itself a realized map (those
    handle the forward dunder). An unmaterialized :class:`Operator`
    on the left raises the guard; anything else yields
    ``NotImplemented``.

    Parameters
    ----------
    inner : RealizedMap
        The inner (applied-first) realized map.
    outer : object
        The outer (applied-last) operand.

    Returns
    -------
    RealizedMap
        Always ``NotImplemented`` unless the guard raises.
    """
    _guard_operator(outer, operation="RealizedMap.__matmul__")
    return NotImplemented


def realized_sum(left: object, right: object) -> RealizedMap:
    """
    Build the flat realized-map sum ``left + right``.

    Description
    -----------
    Flattens nested :class:`RealizedSum` terms, typechecks a common
    signature through the S0a :func:`union_spaces` validator (all
    terms share a ``domain`` and a ``codomain``), and collapses a
    singleton. A non-realized operand yields ``NotImplemented`` (the
    materialization guard fires for a bare operator).

    Parameters
    ----------
    left : object
        The left addend.
    right : object
        The right addend.

    Returns
    -------
    RealizedMap
        The flattened sum (or ``NotImplemented``).
    """
    if not (isinstance(left, RealizedMap)
            and isinstance(right, RealizedMap)):
        _guard_operator(left, operation="RealizedMap.__add__")
        _guard_operator(right, operation="RealizedMap.__add__")
        return NotImplemented
    terms = (*_sum_terms(left), *_sum_terms(right))
    domain = terms[0].domain
    codomain = terms[0].codomain
    for term in terms[1:]:
        union_spaces(domain, term.domain,
                     operation="RealizedMap.__add__")
        union_spaces(codomain, term.codomain,
                     operation="RealizedMap.__add__")
    if len(terms) == 1:
        return terms[0]
    return RealizedSum(terms)


def compose_realized(
    outer: RealizedMap, inner: RealizedMap,
) -> RealizedMap:
    """
    Flatten, eager-fuse, typecheck, and build ``outer @ inner``.

    Description
    -----------
    The realized-map composition core: nested composites flatten
    (associativity), adjacent ``Symbol`` factors **eager-fuse** to a
    single ``Symbol`` (preserving the bitwise Hadamard fast path — a
    homogeneous ``Symbol`` chain never lands in a composite), each
    surviving adjacency is typechecked through the S0a
    :func:`compose_spaces` validator, and a lone survivor is returned
    unwrapped.

    Parameters
    ----------
    outer : RealizedMap
        The outer (applied-last) map.
    inner : RealizedMap
        The inner (applied-first) map.

    Returns
    -------
    RealizedMap
        The fused ``Symbol`` or the flat ``RealizedComposite``.
    """
    factors = [*_chain_factors(outer), *_chain_factors(inner)]
    factors = _fuse_adjacent_symbols(factors)
    for above, below in itertools.pairwise(factors):
        compose_spaces(below.domain, below.codomain,
                       above.domain, above.codomain)
    if len(factors) == 1:
        return factors[0]
    return RealizedComposite(tuple(factors))


def _chain_factors(m: RealizedMap) -> list[RealizedMap]:
    """Flatten a composite into its factors (others pass through)."""
    if isinstance(m, RealizedComposite):
        return list(m.factors)
    return [m]


def _sum_terms(m: RealizedMap) -> tuple[RealizedMap, ...]:
    """Flatten a sum into its terms (others pass through)."""
    if isinstance(m, RealizedSum):
        return m.terms
    return (m,)


def _fuse_adjacent_symbols(
    factors: list[RealizedMap],
) -> list[RealizedMap]:
    """Collapse runs of adjacent ``Symbol`` factors via ``@``."""
    fused: list[RealizedMap] = []
    for factor in factors:
        if fused and isinstance(fused[-1], Symbol) and isinstance(
                factor, Symbol):
            fused[-1] = fused[-1] @ factor
        else:
            fused.append(factor)
    return fused


# ================================================================
#  The lazy composite
# ================================================================
@partial(fr.utils.jaxify, dynamic=("_factors",))
class RealizedComposite:

    r"""
    A lazy, grid-bound chain of realized maps (right-to-left apply).

    Description
    -----------
    The realized-map twin of the operator :class:`Composite`
    (``base.py``): an ordered tuple of factors, outermost
    (applied-last) first, so ``codomain`` reads ``factors[0]`` and
    ``domain`` reads ``factors[-1]``. Applying threads the field
    right-to-left through each factor's ``__call__`` (the
    ``Composite._apply`` pattern). Built only through ``@`` (via
    :func:`compose_realized`, which flattens and eager-fuses adjacent
    symbols), never nested and never wrapping a lone fused ``Symbol``.

    The factor tuple is the single dynamic pytree child: flattening
    recurses into each factor (a ``Symbol``'s ``_data`` leaf, a
    ``BoundTransform``'s static structure), so the composite is jit-
    and ``lax.scan``-friendly.

    Parameters
    ----------
    factors : tuple[RealizedMap, ...]
        The flat factor chain, outermost (applied-last) first. Build
        through ``@`` instead of the plumbing constructor.
    """

    def __init__(self, factors: tuple[RealizedMap, ...]) -> None:
        """Plumbing constructor: build chains with ``@`` instead."""
        self._factors: tuple[RealizedMap, ...] = tuple(factors)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def factors(self) -> tuple[RealizedMap, ...]:
        """The flat factor chain, outermost (applied-last) first."""
        return self._factors

    @property
    def domain(self) -> SpaceLike:
        """Domain of the chain (the innermost factor's domain)."""
        return self._factors[-1].domain

    @property
    def codomain(self) -> SpaceLike:
        """Codomain of the chain (the outermost factor's codomain)."""
        return self._factors[0].codomain

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply the factors right-to-left (each application syncs).

        Parameters
        ----------
        f : FieldLike
            The operand field, on the chain's ``domain``.

        Returns
        -------
        FieldLike
            The result field, on the chain's ``codomain``.
        """
        for op in reversed(self._factors):
            f = op(f)
        return f

    # ================================================================
    #  Algebra
    # ================================================================
    def __matmul__(self, other: object) -> RealizedMap:
        """Compose ``self @ other`` (flatten, fuse, typecheck)."""
        return realized_matmul(self, other)

    def __rmatmul__(self, other: object) -> RealizedMap:
        """Reflected ``other @ self`` (materialization guard)."""
        return realized_rmatmul(self, other)

    def __add__(self, other: object) -> RealizedMap:
        """Sum ``self + other`` (flat, common-signature)."""
        return realized_sum(self, other)

    def __radd__(self, other: object) -> RealizedMap:
        """Reflected sum ``other + self``."""
        return realized_sum(other, self)

    def inverse(self, where_zero: complex = 0.0) -> RealizedComposite:
        r"""
        Reverse the chain and invert each factor.

        Description
        -----------
        ``(A @ B).inverse() == B.inverse() @ A.inverse()``: the factor
        order reverses and each factor is inverted in place (a
        ``Symbol`` pseudo-inverts with ``where_zero``, a
        ``BoundTransform`` swaps direction). No adjacency is created,
        so no re-fusion is needed.

        Parameters
        ----------
        where_zero : complex, optional
            The nullspace gauge threaded to each factor's ``inverse``
            (default: 0.0).

        Returns
        -------
        RealizedComposite
            The inverted chain.
        """
        return RealizedComposite(tuple(
            op.inverse(where_zero) for op in reversed(self._factors)))

    def conj(self) -> RealizedComposite:
        r"""
        Conjugate each factor (order preserved: a homomorphism).

        Description
        -----------
        Conjugation of a linear map is a homomorphism of composition,
        ``conj(A @ B) == conj(A) @ conj(B)``, so the factor order is
        kept. A factor without a defined conjugate (a
        ``BoundTransform`` in iteration 1) raises.

        Returns
        -------
        RealizedComposite
            The conjugated chain.
        """
        return RealizedComposite(tuple(
            op.conj() for op in self._factors))


# ================================================================
#  The lazy sum
# ================================================================
@partial(fr.utils.jaxify, dynamic=("_terms",))
class RealizedSum:

    r"""
    A lazy, grid-bound sum of realized maps on a common signature.

    Description
    -----------
    The realized-map twin of :class:`OperatorSum`: a flat term list,
    all sharing a ``domain`` and a ``codomain`` (validated on build
    through :func:`union_spaces`), applied by summing each term's
    result. Built only through ``+`` (:func:`realized_sum`, which
    flattens nested sums). A sum has no closed-form ``inverse``.

    Parameters
    ----------
    terms : tuple[RealizedMap, ...]
        The flat term list. Build through ``+`` instead of the
        plumbing constructor.
    """

    def __init__(self, terms: tuple[RealizedMap, ...]) -> None:
        """Plumbing constructor: build sums with ``+`` instead."""
        self._terms: tuple[RealizedMap, ...] = tuple(terms)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def terms(self) -> tuple[RealizedMap, ...]:
        """The flat term list."""
        return self._terms

    @property
    def domain(self) -> SpaceLike:
        """The shared domain of every term."""
        return self._terms[0].domain

    @property
    def codomain(self) -> SpaceLike:
        """The shared codomain of every term."""
        return self._terms[0].codomain

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply every term to ``f`` and add the (synced) results.

        Parameters
        ----------
        f : FieldLike
            The operand field, on the sum's ``domain``.

        Returns
        -------
        FieldLike
            The summed result, on the sum's ``codomain``.
        """
        total = self._terms[0](f)
        for term in self._terms[1:]:
            total = total + term(f)
        return total

    # ================================================================
    #  Algebra
    # ================================================================
    def __matmul__(self, other: object) -> RealizedMap:
        """Compose ``self @ other`` (the sum is an opaque factor)."""
        return realized_matmul(self, other)

    def __rmatmul__(self, other: object) -> RealizedMap:
        """Reflected ``other @ self`` (materialization guard)."""
        return realized_rmatmul(self, other)

    def __add__(self, other: object) -> RealizedMap:
        """Sum ``self + other`` (flatten nested sums)."""
        return realized_sum(self, other)

    def __radd__(self, other: object) -> RealizedMap:
        """Reflected sum ``other + self``."""
        return realized_sum(other, self)

    def inverse(self, where_zero: complex = 0.0) -> RealizedMap:
        """
        Raise; a sum of realized maps has no closed-form inverse.

        Parameters
        ----------
        where_zero : complex, optional
            Unused; present for the uniform ``inverse`` signature
            (default: 0.0).

        Returns
        -------
        RealizedMap
            Never returns.
        """
        raise NotImplementedError(
            "a sum of realized maps has no closed-form inverse")

    def conj(self) -> RealizedSum:
        """
        Conjugate every term (conjugation distributes over ``+``).

        Returns
        -------
        RealizedSum
            The conjugated sum.
        """
        return RealizedSum(tuple(t.conj() for t in self._terms))


# ================================================================
#  The bound-transform adapter
# ================================================================
@fr.utils.jaxify
class BoundTransform:

    r"""
    A grid-bound transform direction as a fixed-tag realized map.

    Description
    -----------
    The adapter that lets a :class:`Transform` — grid-bound but
    *space-polymorphic* (its ``codomain`` is a resolver, not a fixed
    tag) — participate in the realized-map category. Binding a
    direction and an operand ``space`` resolves the fixed
    ``(domain, codomain)`` pair (``forward`` lands on
    ``transform.codomain(space)``, ``backward`` on
    ``transform.backward_space(space)``). ``inverse`` swaps direction;
    applying delegates to ``forward`` / ``backward``. With this, the
    spectral solve ``backward @ symbol.inverse() @ forward`` composes
    into a :class:`RealizedComposite`.

    Conjugation is designed-for (a transform's conjugate kernel lands
    with the S2 spectral rewrite); the adapter carries no dynamic
    leaves (the transform's plans are static structure).

    Parameters
    ----------
    transform : Transform | ComposedTransform
        The grid-bound transform to adapt (a per-family transform or
        the mixed-product composition).
    space : SpaceLike
        The operand space of this direction (the ``domain``): a nodal
        space for ``forward``, a coefficient space for ``backward``.
    backward : bool, optional
        Whether this adapter is the backward direction
        (coefficient -> nodal); ``False`` is forward (default: False).
    """

    def __init__(
        self,
        transform: Transform | ComposedTransform,
        space: SpaceLike,
        *,
        backward: bool = False,
    ) -> None:
        """Bind the direction and resolve the fixed ``(domain, codomain)``."""
        bare = space.bare
        self._transform: Transform | ComposedTransform = transform
        self._backward: bool = backward
        self._domain: SpaceLike = bare
        self._codomain: SpaceLike = (
            transform.backward_space(bare) if backward
            else transform.codomain(bare))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def transform(self) -> Transform | ComposedTransform:
        """The adapted grid-bound transform."""
        return self._transform

    @property
    def is_backward(self) -> bool:
        """Whether this adapter is the backward direction."""
        return self._backward

    @property
    def domain(self) -> SpaceLike:
        """The fixed domain of this direction."""
        return self._domain

    @property
    def codomain(self) -> SpaceLike:
        """The fixed codomain of this direction."""
        return self._codomain

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply the bound direction (``forward`` or ``backward``).

        Parameters
        ----------
        f : FieldLike
            The operand field, on this adapter's ``domain``.

        Returns
        -------
        FieldLike
            The transformed field, on this adapter's ``codomain``.
        """
        if self._backward:
            return self._transform.backward(f)
        return self._transform.forward(f)

    # ================================================================
    #  Algebra
    # ================================================================
    def __matmul__(self, other: object) -> RealizedMap:
        """Compose ``self @ other`` (flatten, fuse, typecheck)."""
        return realized_matmul(self, other)

    def __rmatmul__(self, other: object) -> RealizedMap:
        """Reflected ``other @ self`` (materialization guard)."""
        return realized_rmatmul(self, other)

    def __add__(self, other: object) -> RealizedMap:
        """Sum ``self + other`` (common-signature)."""
        return realized_sum(self, other)

    def __radd__(self, other: object) -> RealizedMap:
        """Reflected sum ``other + self``."""
        return realized_sum(other, self)

    def inverse(
        self,
        where_zero: complex = 0.0,  # noqa: ARG002 — a transform is exact
    ) -> BoundTransform:
        """
        Return the opposite direction (``forward`` <-> ``backward``).

        Parameters
        ----------
        where_zero : complex, optional
            Unused; present for the uniform ``inverse`` signature (a
            transform is exactly invertible, default: 0.0).

        Returns
        -------
        BoundTransform
            The reverse-direction adapter.
        """
        return BoundTransform(self._transform, self._codomain,
                              backward=not self._backward)

    def conj(self) -> BoundTransform:
        """
        Conjugate of a bound transform: designed-for (S2+).

        Returns
        -------
        BoundTransform
            Never returns.
        """
        raise NotImplementedError(
            "the conjugate of a bound transform is designed-for; it "
            "lands with the S2 spectral rewrite")
