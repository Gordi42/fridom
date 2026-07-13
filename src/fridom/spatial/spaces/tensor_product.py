"""
``TensorProductSpace`` and the strict-algebra join.

Description
-----------
Owning class doc: ``design/specs/grid/classes/product_spaces.md``.
This module owns the flat, associative, weakly interned product of
per-mesh factor spaces, the ``SpaceLike`` alias, and the shared
semantics of binary field arithmetic: the two sanctioned per-factor
lifts (constant broadcast, real → complex promotion), the layout
equality check, and the join (per-factor least upper bound). Fields
(Wave 2) call ``join`` / ``require_same_layout``; everything here is
identity comparison plus a few ``is``-checks per factor — cheap and
jit-static.
"""
# Wave 1: TensorProductSpace, SpaceLike, join machinery
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.interning import InternTable
from fridom.spatial.scalars import Scalars, Variance
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.function_space import FunctionSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from fridom.spatial.decomposition.layout import Layout


class TensorProductSpace:

    """
    Flat, interned product of factor spaces, one per mesh.

    Description
    -----------
    The space a multi-dimensional field lives on. Construction goes
    through ``of`` (or ``*``), which normalizes to a flat tuple of
    per-mesh factors and interns the result in a weak table, so
    ``a * (b * c) is (a * b) * c``. ``__eq__`` / ``__hash__`` are
    deliberately not overridden: interning makes equality an
    identity comparison. Products hold no grid reference and no
    arrays; they are pure static keys.

    Parameters
    ----------
    factors : tuple[FunctionSpace, ...]
        The flat per-mesh factor spaces (bare, coordinate order).
    layout : Layout | None, optional
        The negotiated device layout; None for a bare product
        (default: None).
    variance : Variance | None, optional
        The component variance of the whole product (validation
        section 6.3); None means scalar/no-variance
        (default: None).
    """

    # class-level weak intern table keyed on the factor-id tuple
    # (plus the layout/variance when set): unreferenced products are
    # collected, meshes are not pinned, no state leaks across tests
    _table: ClassVar[InternTable] = InternTable()

    def __init__(self, factors: tuple[FunctionSpace, ...],
                 *, layout: Layout | None = None,
                 variance: Variance | None = None) -> None:
        """Plumbing constructor; not interned — use ``of`` or ``*``."""
        self._factors: tuple[FunctionSpace, ...] = tuple(factors)
        self._layout: Layout | None = layout
        self._variance: Variance | None = variance

    # ================================================================
    #  Interned construction
    # ================================================================
    @classmethod
    def _intern(cls, factors: tuple[FunctionSpace, ...],
                layout: Layout | None,
                variance: Variance | None = None,
                ) -> TensorProductSpace:
        """Return the interned (factors, layout, variance) product."""
        key: tuple = factors
        if layout is not None:
            key = (*key, ("layout", layout))
        if variance is not None:
            key = (*key, ("variance", variance))
        return cls._table.intern(
            key,
            lambda: cls(factors, layout=layout, variance=variance))

    @classmethod
    def of(
        cls, *spaces: FunctionSpace | TensorProductSpace,
    ) -> FunctionSpace | TensorProductSpace:
        """
        Normalize, validate, and intern a product of spaces.

        Description
        -----------
        Nesting is collapsed (a product argument contributes its
        factors, never itself); factors keep left-to-right order;
        duplicate coordinate names and the empty product are
        rejected (``ValueError``); a single-factor request returns
        the factor itself — the product of one thing is that thing.
        ``of`` mints bare products only: laid-out variants are
        grid-minted through ``with_layout``.

        Parameters
        ----------
        *spaces : FunctionSpace | TensorProductSpace
            The factor spaces and/or products to combine.

        Returns
        -------
        FunctionSpace | TensorProductSpace
            The interned flat product, or the lone factor itself.
        """
        if len(spaces) == 1 and isinstance(
                spaces[0], FunctionSpace | TensorProductSpace):
            return spaces[0]
        factors = _flatten_bare(spaces)
        if not factors:
            raise ValueError(
                "the empty product is rejected; 'trivial along mesh "
                "m' is spelled per mesh as m.constant")
        _reject_duplicate_names(factors)
        return cls._intern(factors, None)

    def __mul__(
        self, other: FunctionSpace | TensorProductSpace,
    ) -> TensorProductSpace:
        """Tensor product (flat, associative): ``self ⊗ other``."""
        return type(self).of(self, other)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def factors(self) -> tuple[FunctionSpace, ...]:
        """The per-mesh factor spaces (bare, flat, in order)."""
        return self._factors

    @property
    def names(self) -> tuple[str, ...]:
        """All coordinate names, concatenated across factors."""
        return tuple(
            name for factor in self._factors for name in factor.names)

    @property
    def active_axis_names(self) -> tuple[str, ...]:
        """
        Coordinate names of the non-constant factors, flattened.

        Description
        -----------
        The public "active axis names" accessor: the coordinate
        names of the factors that are not constant/broadcast
        factors — the names a separable operator can bind against.

        Returns
        -------
        tuple[str, ...]
            The coordinate names of the non-constant factors.
        """
        return tuple(
            name
            for factor in self._factors
            if not factor.is_constant
            for name in factor.names)

    @property
    def has_constant_factor(self) -> bool:
        """Whether any factor is the constant/broadcast factor."""
        return any(factor.is_constant for factor in self._factors)

    @property
    def shape(self) -> tuple[int, ...]:
        """True global DOF shape: concatenated factor shapes."""
        return tuple(
            n for factor in self._factors for n in factor.shape)

    @property
    def ndim(self) -> int:
        """Number of array axes, ``len(self.shape)``."""
        return len(self.shape)

    @property
    def scalars(self) -> Scalars:
        """The Körper of the product (derived, not stored).

        ``fr.Complex`` if any factor is complex, else ``fr.Real``.
        """
        for factor in self._factors:
            if factor.scalars is Scalars.COMPLEX:
                return Scalars.COMPLEX
        return Scalars.REAL

    @property
    def layout(self) -> Layout | None:
        """Negotiated device layout, or None for a bare space."""
        return self._layout

    @property
    def variance(self) -> Variance | None:
        """Component variance of the whole product, or None.

        Covariant/contravariant products are distinct interned
        spaces (validation section 6.3); factors stay untagged —
        the tag lives on the component's full space.
        """
        return self._variance

    @property
    def bare(self) -> TensorProductSpace:
        """The layout-free interned variant (self if bare).

        The variance tag is part of the bare identity and is kept.
        """
        if self._layout is None:
            return self
        return self._intern(self._factors, None, self._variance)

    def with_layout(
        self, layout: Layout | None,
    ) -> TensorProductSpace:
        """
        Return the interned variant carrying ``layout``.

        Parameters
        ----------
        layout : Layout | None
            The device layout; None returns the bare variant.

        Returns
        -------
        TensorProductSpace
            The interned variant (grid-minted in normal operation).
        """
        return self._intern(self._factors, layout, self._variance)

    def with_variance(
        self, variance: Variance | None,
    ) -> TensorProductSpace:
        """
        Return the interned variant carrying ``variance``.

        Description
        -----------
        The product-space twin of
        ``FunctionSpace.with_variance``: the tag is an attribute of
        the component's *whole* space — the factors stay untagged,
        so per-factor dispatch (``("diff", factor)``, ...) is
        untouched by tagging. ``None`` strips the tag.

        Parameters
        ----------
        variance : Variance | None
            The component variance; None returns the untagged
            variant.

        Returns
        -------
        TensorProductSpace
            The interned variant with the requested variance.
        """
        if variance is not None and not isinstance(variance,
                                                   Variance):
            raise TypeError(
                f"variance must be a Variance member or None, got "
                f"{variance!r}")
        return self._intern(self._factors, self._layout, variance)

    # ================================================================
    #  Factor access and derived products
    # ================================================================
    def factor(self, name: str) -> FunctionSpace:
        """
        Return the factor space contributing coordinate ``name``.

        Parameters
        ----------
        name : str
            A coordinate name of one of the factors.

        Returns
        -------
        FunctionSpace
            The (bare) factor carrying ``name``.
        """
        for factor in self._factors:
            if name in factor.names:
                return factor
        raise KeyError(
            f"no factor contributes coordinate {name!r}; this "
            f"product's names are {self.names}")

    def replace(
        self, **by_name: FunctionSpace,
    ) -> TensorProductSpace:
        """
        Return a new interned product with named factors substituted.

        Description
        -----------
        The codomain-building primitive for operators and
        reductions. Substituting via *any* name of a multi-name
        factor replaces that whole factor; the replacement must live
        on the same mesh (``ValueError`` otherwise) and be bare. The
        layout is preserved.

        Parameters
        ----------
        **by_name : FunctionSpace
            Mapping of coordinate name to the new factor space.

        Returns
        -------
        TensorProductSpace
            The interned substituted product.
        """
        factors = list(self._factors)
        for name, replacement in by_name.items():
            index = next(
                (i for i, factor in enumerate(factors)
                 if name in factor.names), None)
            if index is None:
                raise KeyError(
                    f"no factor contributes coordinate {name!r}; "
                    f"this product's names are {self.names}")
            if not isinstance(replacement, FunctionSpace):
                raise TypeError(
                    f"replacement for {name!r} must be a factor "
                    f"space, got {replacement!r}")
            if replacement.mesh is not factors[index].mesh:
                raise ValueError(
                    f"replacement for {name!r} must live on the same "
                    "mesh as the factor it replaces")
            factors[index] = replacement.bare
        return self._intern(tuple(factors), self._layout,
                            self._variance)

    def as_complex(self) -> TensorProductSpace:
        """Return the product of complexified factors.

        ``factor.as_complex()`` for all factors; layout and variance
        preserved.
        """
        return self._intern(
            tuple(factor.as_complex() for factor in self._factors),
            self._layout, self._variance)

    def __iter__(self) -> Iterator[FunctionSpace]:
        """Iterate over the factor spaces."""
        return iter(self._factors)

    def __len__(self) -> int:
        """Return the number of factors (meshes), not names."""
        return len(self._factors)

    def __repr__(self) -> str:
        """Render as ``Center(x) ⊗ Center(y)``."""
        base = " ⊗ ".join(repr(factor) for factor in self._factors)
        if self._variance is Variance.COVARIANT:
            base = f"{base} [cov]"
        elif self._variance is Variance.CONTRAVARIANT:
            base = f"{base} [con]"
        if self._layout is None:
            return base
        return f"{base} [layout={self._layout!r}]"


def _flatten_bare(
    spaces: tuple[FunctionSpace | TensorProductSpace, ...],
) -> tuple[FunctionSpace, ...]:
    """Collapse products into their factors; require bare inputs."""
    factors: list[FunctionSpace] = []
    for space in spaces:
        if not isinstance(space,
                          FunctionSpace | TensorProductSpace):
            raise TypeError(
                f"tensor products combine function spaces, got "
                f"{space!r}")
        if space.layout is not None:
            raise ValueError(
                "products are minted bare: strip the layout "
                "(space.bare) before combining laid-out spaces")
        if space.variance is not None:
            raise ValueError(
                "products are minted variance-free: the tag lives "
                "on the component's whole space — combine untagged "
                "factors and tag the product (with_variance)")
        if isinstance(space, TensorProductSpace):
            factors.extend(space.factors)
        else:
            factors.append(space)
    return tuple(factors)


def _reject_duplicate_names(
    factors: tuple[FunctionSpace, ...],
) -> None:
    """Reject duplicate coordinate names (one factor per mesh)."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for factor in factors:
        for name in factor.names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
    if duplicates:
        raise ValueError(
            "duplicate coordinate names in tensor product: "
            f"{tuple(duplicates)} (one factor per mesh)")


# Consumers type against this union; a lone factor space is usable
# wherever a product is expected (shared product protocol).
SpaceLike = FunctionSpace | TensorProductSpace


# ================================================================
#  Shared semantics: lifts, joins, and the strict algebra
# ================================================================
def require_same_layout(
    a: SpaceLike, b: SpaceLike, *, operation: str | None = None,
) -> None:
    """
    Raise unless both spaces carry the same layout.

    Description
    -----------
    The join requires layout equality; the sanctioned lifts never
    touch it, and no implicit reshard exists in field arithmetic.

    Parameters
    ----------
    a : SpaceLike
        The left operand's space.
    b : SpaceLike
        The right operand's space.
    operation : str | None, optional
        The offending operation, for the error message
        (default: None).
    """
    if a.layout == b.layout:
        return
    prefix = "same space, " if a.bare is b.bare else ""
    raise SpaceMismatchError(
        f"{prefix}layouts differ: {a.layout!r} vs {b.layout!r}; use "
        ".reshard(...) — no implicit reshard exists in field "
        "arithmetic",
        left=a, right=b, operation=operation)


def join_factor(
    a: FunctionSpace, b: FunctionSpace,
) -> FunctionSpace | None:
    """
    Per-factor least upper bound under the two sanctioned lifts.

    Description
    -----------
    Exactly two implicit conversions exist, both exact: the constant
    broadcast (a ``ConstantSpace`` factor lifts to any factor space
    on the same mesh) and real → complex promotion (a factor lifts
    to its ``as_complex()`` variant). The join is the least upper
    bound: full factor beats ``ConstantSpace``; complex beats real.

    Parameters
    ----------
    a : FunctionSpace
        One (bare) factor.
    b : FunctionSpace
        The other (bare) factor, on the same mesh.

    Returns
    -------
    FunctionSpace | None
        The joined factor, or None when the factors are not related
        by a chain of the two lifts.
    """
    if a is b:
        return a
    if a.mesh is not b.mesh:
        return None
    a_constant = isinstance(a, ConstantSpace)
    b_constant = isinstance(b, ConstantSpace)
    if a_constant and not b_constant:
        base = b
    elif b_constant and not a_constant:
        base = a
    elif a.as_complex() is b.as_complex():
        # same space up to the Körper (covers constant-constant)
        base = a
    else:
        return None
    want_complex = (a.scalars is Scalars.COMPLEX
                    or b.scalars is Scalars.COMPLEX)
    return base.as_complex() if want_complex else base


def lifts_to(a: FunctionSpace, b: FunctionSpace) -> bool:
    """
    Whether factor ``a`` lifts to factor ``b`` by sanctioned lifts.

    Parameters
    ----------
    a : FunctionSpace
        The factor to lift.
    b : FunctionSpace
        The lift target.

    Returns
    -------
    bool
        True when ``b`` is reachable from ``a`` by a chain of the
        constant broadcast and real → complex promotion (including
        ``a is b``).
    """
    return join_factor(a, b) is b


def join(
    a: SpaceLike, b: SpaceLike, *, operation: str | None = None,
) -> SpaceLike:
    """
    Compute the join of two spaces under the strict algebra.

    Description
    -----------
    Exists iff ``a`` and ``b`` carry the same layout, have factors
    on the same meshes and, per mesh, the factors are identical or
    related by a chain of the two sanctioned lifts. The result is
    the per-factor least upper bound, carrying the common layout.
    Binary field arithmetic (Wave 2) computes this join, lifts both
    operands to it, and applies the operation there. Grid identity
    is checked by the caller *before* the join (fields own that).

    Component variance joins like a claim with a sanctioned lift
    from "no claim" (validation section 6.3): an untagged operand
    adopts the partner's tag (a scalar coefficient scales a tensor
    component without ceremony), while covariant vs contravariant is
    genuine variance mixing and raises — the strict-algebra catch
    the tags exist for.

    Parameters
    ----------
    a : SpaceLike
        The left operand's space.
    b : SpaceLike
        The right operand's space.
    operation : str | None, optional
        The offending operation, for error messages (default: None).

    Returns
    -------
    SpaceLike
        The joined space (a lone factor for 1D operands).
    """
    require_same_layout(a, b, operation=operation)
    variance = _join_variance(a, b, operation=operation)
    if a.variance is not None:
        a = a.with_variance(None)
    if b.variance is not None:
        b = b.with_variance(None)
    factors_a, factors_b = a.factors, b.factors
    suffix = f" ({operation})" if operation else ""
    if len(factors_a) != len(factors_b) or any(
            x.mesh is not y.mesh
            for x, y in zip(factors_a, factors_b, strict=True)):
        mismatched = tuple(sorted(set(a.names) ^ set(b.names)))
        if not mismatched:
            mismatched = a.names
        raise SpaceMismatchError(
            f"cannot combine spaces{suffix}: the factor meshes "
            f"differ ({a!r} vs {b!r})",
            left=a, right=b, operation=operation,
            mismatched_names=mismatched)
    joined: list[FunctionSpace] = []
    diffs: list[str] = []
    mismatched_names: list[str] = []
    for x, y in zip(factors_a, factors_b, strict=True):
        result = join_factor(x, y)
        if result is None:
            mismatched_names.extend(x.names)
            diffs.append(f"{', '.join(x.names)}: {x!r} vs {y!r}")
        else:
            joined.append(result)
    if mismatched_names:
        agreeing = [name for name in a.names
                    if name not in mismatched_names]
        agree = f" ({', '.join(agreeing)} agree)" if agreeing else ""
        raise SpaceMismatchError(
            f"cannot combine spaces{suffix}: {'; '.join(diffs)}"
            f"{agree}; use .to(...) for an explicit conversion",
            left=a, right=b, operation=operation,
            mismatched_names=tuple(mismatched_names))
    result_space: SpaceLike = (
        joined[0] if len(joined) == 1
        else TensorProductSpace.of(*joined))
    if variance is not None:
        result_space = result_space.with_variance(variance)
    if a.layout is not None:
        result_space = result_space.with_layout(a.layout)
    return result_space


def _join_variance(
    a: SpaceLike, b: SpaceLike, *, operation: str | None = None,
) -> Variance | None:
    """
    Join the operands' variance claims (see ``join``).

    Parameters
    ----------
    a : SpaceLike
        The left operand's space.
    b : SpaceLike
        The right operand's space.
    operation : str | None, optional
        The offending operation, for error messages (default: None).

    Returns
    -------
    Variance | None
        The common claim (None when neither operand claims one).
    """
    va, vb = a.variance, b.variance
    if va is vb or vb is None:
        return va
    if va is None:
        return vb
    suffix = f" ({operation})" if operation else ""
    raise SpaceMismatchError(
        f"cannot combine spaces{suffix}: variance mixing "
        f"({a!r} vs {b!r}); raise/lower the index explicitly "
        "(RaiseIndex/LowerIndex) before combining components",
        left=a, right=b, operation=operation)
