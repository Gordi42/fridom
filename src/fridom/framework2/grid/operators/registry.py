"""
``OperatorRegistry``: the layered (kind, space) dispatch table.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_composed.md``
("The operator dispatch registry"). The registry is the dispatch
table behind ``f.diff("x")``, ``f.to(...)``, ``f * g`` and every
other kind (rules 3.4). Its *placement* — the grid owns one as
``grid.dispatch``, seeded with the default entry table — is doc 04's
seam; this module provides the mechanism only. A missing entry is a
``DispatchError`` (a registry error), deliberately distinct from
``SpaceMismatchError`` (a space error).

Two lifecycle refinements land with the grid wiring (grid lifecycle,
operators_composed.md "Transform rows are lazy factories" and the
refined-mesh adoption paragraph): registry values may be
``LazyEntry`` factories — the stored callable builds the grid-bound
operator (a transform) on first ``resolve``, post-negotiation — and
factor lookups fall back along the mesh ``refined_from`` chain, so
spaces on adopted refined meshes resolve the parent mesh's rows with
the same operator instances.
"""
# Wave 2: OperatorRegistry, DispatchError -- Wave 3D: LazyEntry,
#    refined-mesh adoption
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeAlias, final

import fridom.framework as fr
from fridom.framework2.grid.operators.base import (
    Composite,
    Dispatched,
    Operator,
    OperatorRequirements,
    OperatorSum,
    ScaledOperator,
    SeparableComposite,
    resolve_codomain,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import CellAvg, FaceAvg
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.function_space import FunctionSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator, Mapping

    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

#: registry key: kind-only or (kind, space); keys may name a factor
#: space or a full product space (spaces are interned, so keys hash
#: and compare by identity)
DispatchKey: TypeAlias = "str | tuple[str, SpaceLike]"


class DispatchError(KeyError):

    """
    Raised when no operator is registered for (kind, space).

    Description
    -----------
    The registry-resolution failure mode: a missing entry, a
    genuinely mixed product without an exact product key, an unknown
    ``Dispatched`` kind, or a cyclic placeholder resolution. It is
    *not* a space error — operator application to a field outside a
    registered operator's domain raises ``SpaceMismatchError``.
    """


@final
class LazyEntry:

    """
    A lazy registry default: builds its operator on first resolve.

    Description
    -----------
    The mechanism behind the lazily-seeded ``("transform", ...)``
    rows (grid lifecycle step 1): the registry stores this callable
    wrapper, and the grid-bound transform instance is constructed on
    the first ``resolve`` — necessarily post-negotiation, which
    breaks the ``Grid.__init__ -> Fourier(grid) ->
    grid.decomposition`` construction cycle. Construction is
    memoized on the entry, so every key sharing one ``LazyEntry``
    materializes the identical instance (the form-2 product
    resolution requires it). ``items()`` consumers (the halo trace)
    read the *declared* requirements off the entry without
    materializing.

    Parameters
    ----------
    factory : Callable[[], Operator]
        Zero-argument factory building the operator.
    requirements : OperatorRequirements | None, optional
        The declared per-factor requirements reported before
        materialization (default: None, the zero record).
    """

    def __init__(
        self,
        factory: Callable[[], Operator],
        requirements: OperatorRequirements | None = None,
    ) -> None:
        """Store the factory and the declared requirements."""
        if not callable(factory):
            raise TypeError(
                f"lazy entries wrap a zero-argument factory, got "
                f"{factory!r}")
        self._factory: Callable[[], Operator] = factory
        self._requirements: OperatorRequirements = (
            OperatorRequirements() if requirements is None
            else requirements)
        self._instance: Operator | None = None

    def materialize(self) -> Operator:
        """Build (once) and return the wrapped operator."""
        if self._instance is None:
            self._instance = _check_operator(self._factory())
        return self._instance

    def requirements(
        self,
        domain: SpaceLike,  # noqa: ARG002 — declared, not resolved
    ) -> OperatorRequirements:
        """
        Report the declared requirements (no materialization).

        Parameters
        ----------
        domain : SpaceLike
            The factor space the halo trace sizes on.

        Returns
        -------
        OperatorRequirements
            The declared per-factor requirements record.
        """
        return self._requirements


@final
@fr.utils.jaxify
class OperatorRegistry:

    """
    Layered (kind, space) -> Operator dispatch table.

    Description
    -----------
    Entries are array-free operators (G1). Keys are a kind alone
    (fires for every space the kind can fire on) or ``(kind, space)``
    with a factor *or* product space. ``resolve`` applies the
    precedence override ``(kind, space)`` > override kind-only >
    default ``(kind, space)`` > default kind-only; product spaces
    walk the three key forms (exact product key, per-factor fallback,
    kind-only). ``__setitem__`` is setup-time mutation only; after
    assembly the registry is treated as frozen structure (the freeze
    point is enforced by the grid, not this container).

    Parameters
    ----------
    defaults : Mapping[DispatchKey, Operator] | None, optional
        The default entry table (default: None, empty).
    """

    def __init__(
        self,
        defaults: Mapping[DispatchKey, Operator] | None = None,
    ) -> None:
        """Create a registry from a default entry table."""
        entries: dict[DispatchKey, Operator] = {}
        if defaults is not None:
            for key, op in defaults.items():
                entries[_normalize_key(key)] = _check_entry(op)
        #: layered entry dicts, outermost (highest precedence) first
        self._layers: tuple[dict[DispatchKey, Operator], ...] = (
            entries,)

    # ================================================================
    #  Identity (mutable setup-time container; never value-compared)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other``."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    # ================================================================
    #  Mapping surface
    # ================================================================
    def __getitem__(self, key: DispatchKey) -> Operator:
        """
        Exact-entry lookup (no precedence fallback).

        Parameters
        ----------
        key : DispatchKey
            The kind or (kind, space) key.

        Returns
        -------
        Operator
            The outermost layer's exact entry for the key.
        """
        nkey = _normalize_key(key)
        for layer in self._layers:
            if nkey in layer:
                return _materialize(layer[nkey])
        raise DispatchError(f"no exact entry for key {key!r}")

    def __setitem__(self, key: DispatchKey, op: Operator) -> None:
        """
        Register/override an entry (setup-time only, pre-trace).

        Parameters
        ----------
        key : DispatchKey
            The kind or (kind, space) key.
        op : Operator
            The operator (or ``LazyEntry``) to register.
        """
        self._layers[0][_normalize_key(key)] = _check_entry(op)

    def __contains__(self, key: DispatchKey) -> bool:
        """Whether an exact entry exists for the key."""
        nkey = _normalize_key(key)
        return any(nkey in layer for layer in self._layers)

    def items(self) -> Iterator[tuple[DispatchKey, Operator]]:
        """
        Iterate all visible entries (halo-accounting trace consumer).

        Description
        -----------
        Keys are deduplicated with layer precedence: an override
        shadows the default entry under the same key.

        Yields
        ------
        tuple[DispatchKey, Operator]
            The visible (key, operator) pairs, outermost layer first.
        """
        seen: set[DispatchKey] = set()
        for layer in self._layers:
            for key, op in layer.items():
                if key not in seen:
                    seen.add(key)
                    yield key, op

    # ================================================================
    #  Dispatch
    # ================================================================
    def resolve(self, kind: str, space: SpaceLike) -> Operator:
        """
        Resolve with precedence; raise ``DispatchError`` if absent.

        Parameters
        ----------
        kind : str
            The dispatch kind.
        space : SpaceLike
            The factor or product space keying the resolution
            (layouts are stripped: dispatch keys never see them).

        Returns
        -------
        Operator
            The registered operator.
        """
        bare = space.bare
        if isinstance(bare, TensorProductSpace):
            return self._resolve_product(kind, bare)
        op = self._lookup_factor(kind, bare)
        if op is None:
            raise DispatchError(
                f"no operator registered for kind {kind!r} on "
                f"{bare!r}")
        return op

    def merge(
        self, overrides: Mapping[DispatchKey, Operator],
    ) -> OperatorRegistry:
        """
        Pure layered copy with ``overrides`` on top (assembly).

        Description
        -----------
        Returns a new registry; this one is never mutated. Every
        space-keyed entry containing ``Dispatched`` placeholders is
        resolved against the *merged* registry with ordinary
        precedence and stored concrete (D4) — after the merge, no
        per-application registry lookup happens inside chains.
        Kind-only entries keep their holes: they resolve at
        application, when the operand space is known.

        Parameters
        ----------
        overrides : Mapping[DispatchKey, Operator]
            Module override entries to layer on top.

        Returns
        -------
        OperatorRegistry
            The merged registry with concrete space-keyed entries.
        """
        top: dict[DispatchKey, Operator] = {}
        for key, op in overrides.items():
            top[_normalize_key(key)] = _check_entry(op)
        merged = OperatorRegistry()
        merged._layers = (top, *self._layers)
        for key, op in tuple(merged.items()):
            if isinstance(key, tuple) and _contains_hole(op):
                top[key] = _resolve_entry(
                    op, merged, key[1], frozenset())
        return merged

    # ================================================================
    #  Internal lookup machinery
    # ================================================================
    def _lookup_factor(
        self, kind: str, factor: SpaceLike,
    ) -> Operator | None:
        """
        Layer-major lookup: (kind, factor) beats kind-only.

        Description
        -----------
        Space keys walk the factor's refined-mesh **adoption chain**
        (grid.md's normative adoption paragraph): a space on a
        ``mesh.refined(...)`` descendant of a seeded mesh resolves
        the parent's row — the same operator instance — so the
        refined meshes minted by padded transforms after grid
        construction get their product/stencil rows without
        re-seeding.
        """
        for layer in self._layers:
            for candidate in _adoption_chain(factor):
                entry = layer.get((kind, candidate))
                if entry is not None:
                    return _materialize(entry)
            entry = layer.get(kind)
            if entry is not None:
                return _materialize(entry)
        return None

    def _resolve_product(
        self, kind: str, product: TensorProductSpace,
    ) -> Operator:
        """
        Resolve a product-space key through the three key forms.

        Description
        -----------
        Form 1: the exact (interned) product key. Form 2: drop
        ``ConstantSpace`` factors and resolve per factor — all
        factors must yield the **same operator instance**, otherwise
        the product is genuinely mixed and raises. Form 3: the
        kind-only entry. Within each form, overrides beat defaults.

        Parameters
        ----------
        kind : str
            The dispatch kind.
        product : TensorProductSpace
            The bare product space.

        Returns
        -------
        Operator
            The registered operator.
        """
        for layer in self._layers:
            entry = layer.get((kind, product))
            if entry is not None:
                return _materialize(entry)
        factors = tuple(f for f in product.factors
                        if not isinstance(f, ConstantSpace))
        if factors:
            resolved = [self._lookup_factor(kind, f) for f in factors]
            if all(r is not None for r in resolved):
                first = resolved[0]
                if any(r is not first for r in resolved):
                    raise DispatchError(
                        f"kind {kind!r} resolves to different "
                        f"operators across the factors of "
                        f"{product!r}; a genuinely mixed product "
                        "needs an explicit exact product key")
                return first
        for layer in self._layers:
            entry = layer.get(kind)
            if entry is not None:
                return _materialize(entry)
        raise DispatchError(
            f"no operator registered for kind {kind!r} on "
            f"{product!r}")


# ================================================================
#  Key/value validation
# ================================================================
def _normalize_key(key: DispatchKey) -> DispatchKey:
    """Validate a dispatch key and strip layouts off space keys."""
    if isinstance(key, str):
        return key
    if (isinstance(key, tuple) and len(key) == 2  # noqa: PLR2004
            and isinstance(key[0], str)
            and isinstance(key[1],
                           FunctionSpace | TensorProductSpace)):
        return (key[0], key[1].bare)
    raise TypeError(
        f"dispatch keys are a kind or (kind, space), got {key!r}")


def _check_operator(op: Operator) -> Operator:
    """Validate a (materialized) operator value."""
    if not isinstance(op, Operator):
        raise TypeError(
            f"registry entries are operators, got {op!r}")
    return op


def _check_entry(op: Operator | LazyEntry) -> Operator | LazyEntry:
    """Validate a registry value (operator or lazy factory)."""
    if isinstance(op, LazyEntry):
        return op
    return _check_operator(op)


def _materialize(entry: Operator | LazyEntry) -> Operator:
    """Build a lazy entry (memoized); pass operators through."""
    if isinstance(entry, LazyEntry):
        return entry.materialize()
    return entry


def _adoption_chain(factor: SpaceLike) -> Iterator[SpaceLike]:
    """
    Yield the factor, then its parent-mesh siblings (adoption).

    Description
    -----------
    Walks doc 01's ``refined_from`` link: for each refinement parent
    the *same* space family is interned on the parent mesh (nodal
    node set + BC structure, cell/face averages; scalars carried
    over). Families without a parent sibling (coefficient factors,
    constants) end the walk — refined coefficient rows are minted by
    their own transforms.
    """
    yield factor
    current = factor
    while True:
        parent = getattr(current.mesh, "refined_from", None)
        if parent is None:
            return
        if isinstance(current, NodalSpace):
            sibling = parent.nodal(current.node_set, bc=current.bc)
        elif isinstance(current, CellAvg):
            sibling = parent.cell_avg
        elif isinstance(current, FaceAvg):
            sibling = parent.face_avg
        else:
            return
        if current.scalars is Scalars.COMPLEX:
            sibling = sibling.as_complex()
        yield sibling
        current = sibling


# ================================================================
#  Dispatched-placeholder resolution at merge (D4)
# ================================================================
def _contains_hole(op: Operator) -> bool:
    """Whether ``op`` contains a ``Dispatched`` placeholder."""
    if isinstance(op, Dispatched):
        return True
    if isinstance(op, Composite | SeparableComposite):
        return any(_contains_hole(f) for f in op.factors)
    if isinstance(op, OperatorSum):
        return any(_contains_hole(t) for t in op.terms)
    if isinstance(op, ScaledOperator):
        return _contains_hole(op.target)
    return False


def _resolve_entry(
    op: Operator,
    registry: OperatorRegistry,
    space: SpaceLike,
    active: frozenset[str],
) -> Operator:
    """
    Recursively replace ``Dispatched`` holes with registry entries.

    Description
    -----------
    Chains thread the resolution space right-to-left through the
    already-resolved factors' codomains; rebuilt structures go back
    through the algebra dunders, so interning (D6) canonicalizes the
    result. ``active`` guards against cyclic kind references.

    Parameters
    ----------
    op : Operator
        The (possibly hole-carrying) entry operator.
    registry : OperatorRegistry
        The merged registry resolution runs against.
    space : SpaceLike
        The entry key's space (the operand space of the chain).
    active : frozenset[str]
        Kinds currently being resolved (cycle guard).

    Returns
    -------
    Operator
        The fully concrete operator.
    """
    if isinstance(op, Dispatched):
        if op.kind in active:
            raise DispatchError(
                f"cyclic Dispatched resolution through kind "
                f"{op.kind!r}")
        target = registry.resolve(op.kind, space)
        target = _resolve_entry(
            target, registry, space, active | {op.kind})
        if op.bound_axis is not None:
            target = target[op.bound_axis]
        return target
    if isinstance(op, Composite | SeparableComposite):
        resolved: list[Operator] = []
        current = space
        for factor in reversed(op.factors):
            concrete = _resolve_entry(factor, registry, current,
                                      active)
            resolved.append(concrete)
            current = resolve_codomain(concrete, current)
        resolved.reverse()
        chain = functools.reduce(lambda a, b: a @ b, resolved)
        axis = getattr(op, "bound_axis", None)
        return chain if axis is None else chain[axis]
    if isinstance(op, OperatorSum):
        terms = [_resolve_entry(t, registry, space, active)
                 for t in op.terms]
        return functools.reduce(lambda a, b: a + b, terms)
    if isinstance(op, ScaledOperator):
        return _resolve_entry(
            op.target, registry, space, active) * op.coeff
    return op
