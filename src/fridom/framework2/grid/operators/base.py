"""
The ``Operator`` hierarchy and the operator algebra.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_base.md``
(with the decision log ``operator_algebra_merge.md`` as normative
parent). This module owns the base hierarchy (``Operator``,
``UnaryOperator``, ``BinaryOperator``, ``SeparableOperator``,
``OperatorRequirements``, ``EigenbasisError``) and the algebra
objects built by the dunders (``Identity``, ``Zero``, ``Composite``,
``SeparableComposite``, ``OperatorSum``, ``ScaledOperator``,
``Block``, ``Dispatched``).

Operators are static structure: identity-hashed, interned on their
static structure (D6), and free of per-grid arrays. Fields are
consumed through the documented duck surface of
``notes/framework2/classes/fields.md`` only (see ``FieldLike``): the
shared application path reads ``f.function_space``, ``f.grid``
(``grid.sync``, ``grid.dispatch``), ``f.data`` / ``f._data``,
``f.metadata``, ``f.halo_valid``, ``f.with_data``, and rebuilds
fields through the plumbing constructor
``type(f)(grid, space, data, metadata, halo_valid=...)``.
"""
# Wave 2: Operator, UnaryOperator, BinaryOperator,
#    SeparableOperator, OperatorRequirements, EigenbasisError,
#    Identity, Zero, Composite, SeparableComposite, OperatorSum,
#    ScaledOperator, Block, Dispatched
from __future__ import annotations

import contextlib
import copy
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal, Self, TypeAlias, final

import jax

import fridom.framework as fr
from fridom.framework2.grid.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.framework2.grid.interning import InternTable
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.function_space import FunctionSpace
from fridom.framework2.grid.spaces.tensor_product import (
    require_same_layout,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.operators.registry import OperatorRegistry
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

#: Duck-typed field operand: the iteration-1 ``ScalarField`` surface
#: (fields.md) the operator layer is allowed to consume —
#: ``function_space``, ``grid`` (with ``sync`` and ``dispatch``),
#: ``data`` / ``_data`` (true-shape view / storage-shaped array),
#: ``metadata``, ``halo_valid``, ``with_data``, field arithmetic
#: dunders, and the trusting plumbing constructor
#: ``type(f)(grid, function_space, data, metadata, halo_valid=...)``.
FieldLike: TypeAlias = object

# Module-level weak intern table for bound variants and algebra
# objects (D6): keys are static-structure tuples over identity-hashed
# constituents, so structurally-equal requests return the identical
# object and the identity-hash invariant (``__eq__``/``__hash__`` are
# ``self is other``) stays jit-cache friendly.
_ALGEBRA_TABLE = InternTable()


# ================================================================
#  Value types and errors
# ================================================================
@dataclass(frozen=True)
class OperatorRequirements:

    """
    Per-factor decomposition requirements of an operator.

    Description
    -----------
    Frozen value type read by the grid's halo-accounting trace, one
    instance per applied factor. Composites accumulate it along their
    chains (halo sums), sums combine it across terms (halo maxes).

    Parameters
    ----------
    halo : int, optional
        Ghost-layer depth needed along this factor's axis
        (default: 0).
    layout : Literal["any", "local", "transpose"], optional
        "any" works on a sharded axis via halo exchange; "local"
        needs the factor's axis undistributed; "transpose" is
        distributed via transpose-based scheduling (default: "any").
    collective : bool, optional
        Informational: the operator performs a cross-shard reduction;
        negotiation treats it as no-constraint (default: False).
    """

    halo: int = 0
    layout: Literal["any", "local", "transpose"] = "any"
    collective: bool = False


class EigenbasisError(TypeError):

    """Raised when eigenvalues are queried outside a diagonalizing basis."""


# ================================================================
#  Root ABC
# ================================================================
@fr.utils.jaxify
class Operator(ABC):

    """
    Typed map between function spaces; free-standing and grid-free.

    Description
    -----------
    Root of the operator hierarchy (operators_base.md section "Base
    hierarchy"). The signature is a resolver, not stored state:
    ``codomain(*domains)`` maps bare domain space(s) to bare codomain
    space(s); unsupported domains raise ``SpaceMismatchError``. The
    algebra dunders build the shallow eager algebra objects of this
    module — the only normalizations are chain-flattening,
    ``Identity`` elision, and ``Zero`` dropping; there is no
    algebraic rewriting. Eigenvalues and block expansion are optional
    capabilities (B4): the base ``eigenvalues`` raises
    ``EigenbasisError``.
    """

    #: suggested registry kind for convenience registration; the
    #: authoritative kind is always the registry key (rules 3.4)
    dispatch_kind: ClassVar[str | None] = None

    # ================================================================
    #  Identity (cluster rules: identity-hashed static structure)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other`` (interned structure)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    # ================================================================
    #  Signature and requirements
    # ================================================================
    @abstractmethod
    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """
        Resolve the codomain from the (bare) domain space(s).

        Description
        -----------
        Resolvers see and return bare spaces; the shared application
        path re-attaches the domain's layout (section 5.1). A tuple
        of spaces is returned for direct-sum (vector/tensor)
        signatures (designed-for).

        Parameters
        ----------
        *domains : SpaceLike
            The bare domain space(s), one per operand.

        Returns
        -------
        SpaceLike | tuple[SpaceLike, ...]
            The bare codomain space(s).
        """
        ...

    def codomains(self, *domains: SpaceLike) -> tuple[SpaceLike, ...]:
        """
        Codomain resolver that always returns a tuple.

        Description
        -----------
        The always-tuple companion of :meth:`codomain`: single-space
        signatures are wrapped in a length-1 tuple, direct-sum
        (vector/tensor) signatures pass through unchanged. Call sites
        that iterate the per-output codomains use this instead of
        the ``isinstance(x, tuple)`` normalization ``codomain``
        otherwise forces on them.

        Parameters
        ----------
        *domains : SpaceLike
            The bare domain space(s), one per operand.

        Returns
        -------
        tuple[SpaceLike, ...]
            The bare codomain space(s), always as a tuple.
        """
        result = self.codomain(*domains)
        return result if isinstance(result, tuple) else (result,)

    def requirements(
        self,
        domain: SpaceLike,  # noqa: ARG002 — halo-0 default
    ) -> OperatorRequirements:
        """
        Per-factor decomposition requirements (halo, layout).

        Parameters
        ----------
        domain : SpaceLike
            The (factor) space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            Halo 0 / any layout unless a subclass overrides.
        """
        return OperatorRequirements()

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — capability declined here
        space: SpaceLike,
    ) -> object:
        """
        Diagonal symbol relative to a diagonalizing basis.

        Description
        -----------
        Queried per coefficient *factor* space. The base raises: only
        linear operators over a diagonalizing basis carry symbols
        (optional capability, B4). The ``Symbol`` return type is
        designed-for (Wave 3).

        Parameters
        ----------
        grid : object
            The grid mediating wavenumbers and metric measures.
        space : SpaceLike
            The coefficient factor space.

        Returns
        -------
        object
            The diagonal ``Symbol`` (designed-for).
        """
        raise EigenbasisError(
            f"{type(self).__name__} has no eigenvalues on {space!r}: "
            "not a linear operator over a diagonalizing basis")

    # ================================================================
    #  Algebra (operator_algebra_merge.md D1, T2; rules 3.2/3.4/3.5)
    # ================================================================
    def __matmul__(
        self, other: Operator | tuple[Operator, ...],
    ) -> Operator:
        """
        Composition: ``(A @ B)(f) == A(B(f))``.

        Description
        -----------
        Chains flatten (associativity), ``Identity`` elides, ``Zero``
        absorbs. Separable, axis-compatible operands yield an
        interned ``SeparableComposite``, anything else an interned
        whole-space ``Composite`` (D5/D6). Binary-headed composition
        (T2) is designed-for and raises ``NotImplementedError``.

        Parameters
        ----------
        other : Operator | tuple[Operator, ...]
            The inner operator (applied first); a tuple pre-composes
            a binary operator per input (designed-for).

        Returns
        -------
        Operator
            The interned composite (or a normalized factor).
        """
        if isinstance(other, tuple):
            raise NotImplementedError(
                "binary pre-composition P @ (B1, ..., Bn) is "
                "designed-for (T2); it lands with the binary product "
                "operators")
        if not isinstance(other, Operator):
            return NotImplemented
        if isinstance(self, BinaryOperator) or isinstance(
                other, BinaryOperator):
            raise NotImplementedError(
                "binary-headed composites are designed-for (T2); "
                "they land with Convolution")
        return _compose(self, other)

    def __rmatmul__(self, other: object) -> Operator:
        """Reject ``(A1, A2) @ P``: a left tuple has no meaning."""
        raise TypeError(
            f"cannot compose {other!r} @ {self!r}: the left operand "
            "of @ must be an operator — a tuple on the left has no "
            "meaning (one output, T2)")

    def __add__(self, other: Operator) -> Operator:
        """Operator sum ``(A + B)(f) == A(f) + B(f)`` (flat, interned)."""
        if not isinstance(other, Operator):
            return NotImplemented
        return _sum(self, other)

    def __sub__(self, other: Operator) -> Operator:
        """Difference as ``A + (-1) * B`` (no cancellation rewriting)."""
        if not isinstance(other, Operator):
            return NotImplemented
        return _sum(self, -other)

    def __neg__(self) -> Operator:
        """Negation as ``(-1) * A``."""
        return self * -1

    def __mul__(self, c: complex | FieldLike) -> Operator:
        """
        Coefficient scaling ``(c * A)(f) == A(f) * c``.

        Parameters
        ----------
        c : complex | FieldLike
            A scalar, or a field coefficient living on the codomain.

        Returns
        -------
        Operator
            A ``ScaledOperator`` (``self`` when scaling ``Zero``).
            Scalar-coefficient scalings are interned on the
            coefficient value; field coefficients are dynamic leaves
            and stay un-interned.
        """
        if isinstance(c, Operator):
            return NotImplemented
        if isinstance(self, Zero):
            return self
        if isinstance(c, int | float | complex):
            return _ALGEBRA_TABLE.intern(
                ("scaled", c, self), lambda: ScaledOperator(c, self))
        return ScaledOperator(c, self)

    def __rmul__(self, c: complex | FieldLike) -> Operator:
        """Scaling is commutative in the coefficient: ``c * A``."""
        return self.__mul__(c)

    def __pow__(self, n: int) -> Operator:
        """
        Build the ``n``-fold composition chain ``A @ ... @ A``.

        Parameters
        ----------
        n : int
            Non-negative chain length; ``0`` gives ``Identity``.

        Returns
        -------
        Operator
            The interned chain.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(
                f"operator powers take an integer, got {n!r}")
        if n < 0:
            raise ValueError(
                f"operator powers are non-negative chains, got {n}")
        result: Operator = Identity()
        for _ in range(n):
            result = result @ self
        return result

    # ================================================================
    #  Axis binding (default: "nothing to bind")
    # ================================================================
    def __getitem__(self, axis: str) -> Operator:
        """
        Bind a coordinate axis (bind-only naming, D2).

        Description
        -----------
        Whole-space operators have a fixed signature and raise;
        ``SeparableOperator`` (and ``Dispatched``) override.

        Parameters
        ----------
        axis : str
            The coordinate name to bind.

        Returns
        -------
        Operator
            The interned bound variant (in overriding subclasses).
        """
        raise TypeError(
            f"{type(self).__name__} has a fixed signature; nothing "
            "to bind")


# ================================================================
#  Arity ABCs
# ================================================================
class UnaryOperator(Operator, ABC):

    """
    Operator applied to a single field: ``op(f)`` (bind axes first).

    Description
    -----------
    The final ``__call__`` is the shared application path: resolve
    the bare codomain, run the kernel hook ``_apply``, re-attach the
    domain layout, and sync the result through the field's grid
    (iteration-1 contract: every operator application returns a
    synced field). Subclasses implement ``_apply`` only.
    """

    @abstractmethod
    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        Resolve the codomain from the single (bare) domain space.

        Parameters
        ----------
        domain : SpaceLike
            The bare domain space.

        Returns
        -------
        SpaceLike
            The bare codomain space.
        """
        ...

    @final
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Template: validate, ensure validity, delegate, re-attach.

        Description
        -----------
        ``HaloTracer`` operands are intercepted generically (doc 04):
        the tracer records ``requirements(domain).halo`` and returns
        the codomain tracer without touching kernel code. Real
        operands pass the consumption-side sync check (task 1.8):
        the operand is synced iff its ghost validity is below this
        application's per-axis requirement.

        Parameters
        ----------
        f : FieldLike
            The operand field (duck-typed, see module docstring).

        Returns
        -------
        FieldLike
            The result field on the resolved codomain, with the
            domain's layout re-attached and the kernel's
            halo-validity claim stamped.
        """
        # HaloTracer interception hook (decomposition doc, Wave 3)
        trace = getattr(f, "_trace_apply", None)
        if trace is not None:
            return trace(self)
        codomain = resolve_codomain(self, f.function_space)
        f = _ensure_valid(f, _required_halo(self, f.function_space))
        result = self._apply(f)
        return _finalize(f, result, codomain)

    @abstractmethod
    def _apply(self, f: FieldLike) -> FieldLike:
        """
        Apply to real (non-tracer) operands (subclass hook).

        Description
        -----------
        Kernel hook of the shared application path: builds the result
        field on the **bare** resolved codomain via the plumbing
        constructor; layout re-attachment and the post-kernel sync
        are owned by ``__call__``.

        Parameters
        ----------
        f : FieldLike
            The operand field.

        Returns
        -------
        FieldLike
            The result field (bare codomain, possibly unsynced).
        """
        ...


class BinaryOperator(Operator, ABC):

    """
    Operator applied to two (or more) fields: ``op(f, g)``.

    Description
    -----------
    Same final template pattern as ``UnaryOperator``. The variadic
    ``*more`` slot serves the n-ary elementwise operators
    (``Hadamard``, ``Where``); strict binaries reject extra operands
    in their ``codomain``. Concrete ``codomain`` implementations
    apply the two sanctioned implicit lifts (constant broadcast,
    real -> complex promotion) before demanding equality.
    """

    @abstractmethod
    def codomain(
        self, domain_a: SpaceLike, domain_b: SpaceLike,
    ) -> SpaceLike:
        """
        Resolve the codomain from the two (bare) domain spaces.

        Parameters
        ----------
        domain_a : SpaceLike
            The left operand's bare space.
        domain_b : SpaceLike
            The right operand's bare space.

        Returns
        -------
        SpaceLike
            The bare codomain space.
        """
        ...

    @final
    def __call__(
        self, f: FieldLike, g: FieldLike, *more: FieldLike,
    ) -> FieldLike:
        """
        Template: validate/unite operand spaces, delegate, sync.

        Parameters
        ----------
        f : FieldLike
            The first operand.
        g : FieldLike
            The second operand.
        *more : FieldLike
            Extra operands for n-ary elementwise operators.

        Returns
        -------
        FieldLike
            The synced result field.
        """
        operands = (f, g, *more)
        # HaloTracer interception hook (decomposition doc, Wave 3)
        for operand in operands:
            trace = getattr(operand, "_trace_apply_nary", None)
            if trace is not None:
                return trace(self, operands)
        for other in operands[1:]:
            if other.grid is not f.grid:
                raise GridMismatchError(
                    "operands of a binary operator must live on the "
                    "same grid", left=f.grid, right=other.grid,
                    operation=type(self).__name__)
            require_same_layout(
                f.function_space, other.function_space,
                operation=type(self).__name__)
        codomain = self.codomain(
            *(o.function_space.bare for o in operands))
        # consumption-side sync check (task 1.8): every operand,
        # against the whole-space requirement (the n-ary twin of
        # HaloTracer._trace_apply_nary)
        halo = self.requirements(codomain).halo
        if halo:
            required = {name: halo
                        for factor in codomain.factors
                        if not isinstance(factor, ConstantSpace)
                        for name in factor.names}
            operands = tuple(_ensure_valid(o, required)
                             for o in operands)
        f, g, *more = operands
        result = self._apply(f, g, *more)
        return _finalize(f, result, codomain)

    @abstractmethod
    def _apply(
        self, f: FieldLike, g: FieldLike, *more: FieldLike,
    ) -> FieldLike:
        """
        Apply to real (non-tracer) operands (subclass hook).

        Parameters
        ----------
        f : FieldLike
            The first operand.
        g : FieldLike
            The second operand.
        *more : FieldLike
            Extra operands for n-ary elementwise operators.

        Returns
        -------
        FieldLike
            The result field (bare codomain, possibly unsynced).
        """
        ...


# ================================================================
#  Separable kernels
# ================================================================
class SeparableOperator(UnaryOperator, ABC):

    """
    Separable 1D kernel lifted per factor: ``kernel ⊗ identity``.

    Description
    -----------
    The shape of almost every concrete stencil operator. Binding is
    the only way to name an axis (D2): ``op["x"]`` returns the
    interned bound variant (D6); the inherited ``__call__`` resolves
    the axis in ``_apply`` (``bound_axis`` if set, else the operand's
    sole bindable factor, else a ``ValueError``). Application along a
    ``ConstantSpace`` factor is the identity (rules 3.3). Concrete
    kernels implement the **per-factor** resolver ``codomain``
    (1D factor -> 1D factor) and the kernel hook ``_apply_factor``;
    the shared path (``resolve_codomain``) lifts the per-factor
    signature over bound product spaces.
    """

    #: bound coordinate axis; None = unbound (axis-agnostic 1D
    #: kernel). Part of the STATIC structure (jit/dispatch identity).
    bound_axis: str | None = None

    @final
    def __getitem__(self, axis: str) -> Self:
        """
        Bind to a coordinate axis: an interned static variant.

        Description
        -----------
        Rebinding an already-bound kernel raises (D5/D6).

        Parameters
        ----------
        axis : str
            The coordinate name to bind.

        Returns
        -------
        Self
            The interned bound variant.
        """
        if not isinstance(axis, str):
            raise TypeError(
                f"axes are bound by coordinate name, got {axis!r}")
        if self.bound_axis is not None:
            raise TypeError(
                f"operator is already bound to {self.bound_axis!r}; "
                "rebinding is not allowed")
        return self._rebind(axis)

    @property
    def unbound(self) -> Self:
        """The unbound structural original (``self`` when unbound)."""
        return getattr(self, "_unbound", self)

    def _rebind(self, axis: str) -> Self:
        """
        Interned structural copy with ``bound_axis`` set.

        Parameters
        ----------
        axis : str
            The coordinate name to bind.

        Returns
        -------
        Self
            The interned bound variant.
        """
        base_op = self.unbound

        def build() -> Self:
            clone = copy.copy(base_op)
            clone.__dict__["bound_axis"] = axis
            clone.__dict__["_unbound"] = base_op
            return clone

        return _ALGEBRA_TABLE.intern(("bound", base_op, axis), build)

    def _apply(self, f: FieldLike) -> FieldLike:
        """
        Resolve the axis and run the 1D kernel along it.

        Description
        -----------
        Axis = ``bound_axis``, else the operand's sole bindable
        (non-``ConstantSpace``) factor, else raise. Application along
        a ``ConstantSpace`` factor is the identity.

        Parameters
        ----------
        f : FieldLike
            The operand field.

        Returns
        -------
        FieldLike
            The result of ``_apply_factor`` (or ``f`` unchanged).
        """
        space = f.function_space.bare
        axis = _resolve_axis(self, space)
        if isinstance(space.factor(axis), ConstantSpace):
            return f
        return self._apply_factor(f, axis)

    @abstractmethod
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Per-factor signature: 1D factor -> 1D factor.

        Description
        -----------
        Reused across meshes (e.g. ``Center -> Right`` on a periodic
        mesh). The bound product form (full product -> product with
        the bound factor transformed, identity elsewhere) is realized
        by the shared path (``resolve_codomain``), not by concrete
        kernels.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The bare 1D codomain factor.
        """
        ...

    # ------------------------------------------------------------
    #  Extension contract (subclass hook)
    # ------------------------------------------------------------
    @abstractmethod
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Apply the 1D kernel along the (resolved) factor axis.

        Description
        -----------
        Slice-based over halo-extended storage, never roll-based
        (rules 3.5); spacing enters through the grid's measure fields
        at trace time. Builds the result field on the bare codomain.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis (always a string).

        Returns
        -------
        FieldLike
            The result field.
        """
        ...


# ================================================================
#  Algebra objects: neutral elements
# ================================================================
@final
class Identity(Operator):

    """
    Neutral element of ``@`` (``A @ Identity == A``).

    Description
    -----------
    Elided on chain normalization; what an unbound axis contributes
    in the ``⊗ identity`` extension and what operators along a
    ``ConstantSpace`` axis reduce to. Halo 0. A singleton (D6).
    """

    _instance: ClassVar[Identity | None] = None

    def __new__(cls) -> Identity:
        """Return the interned singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Domains unchanged."""
        return domains[0] if len(domains) == 1 else domains

    def __call__(self, f: FieldLike) -> FieldLike:
        """Return the operand unchanged (already synced)."""
        return f

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the ones diagonal needs no grid
        space: SpaceLike,
    ) -> object:
        """
        Return the all-ones diagonal (neutral element of ``@``).

        Description
        -----------
        The identity map is the unit of the diagonal algebra: a
        ``Symbol`` on ``space.bare`` with ``codomain == space`` whose
        diagonal is one everywhere. A scalar-broadcast ``ones(())``
        suffices — ``Symbol`` multiplies by size-1 broadcasting, so
        ``ScaledOperator(Identity, c)`` yields the ``c`` constant
        symbol and a chain ``A @ Identity`` reduces to ``A``.

        Parameters
        ----------
        grid : object
            The grid (unused: the ones diagonal needs no wavenumbers).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The all-ones diagonal on ``space.bare``.
        """
        from fridom.framework2.grid.operators.symbol import (  # noqa: PLC0415
            Symbol,
        )
        from fridom.framework2.grid.spaces.tensor_product import (  # noqa: PLC0415
            TensorProductSpace,
        )
        # The identity is diagonal in every basis, so its symbol is the
        # all-``Constant`` ones — the universal broadcast wildcard that
        # composes with any coefficient tag (``A @ Identity == A``),
        # rather than a nodal ``space.bare`` that would clash with a
        # neighbour's ``Fourier`` factor in a chain.
        bare = space.bare
        const = TensorProductSpace.of(
            *(factor.mesh.constant for factor in bare.factors))
        return Symbol(const, jax.numpy.ones(()))


@final
class Zero(Operator):

    """
    Neutral of ``+`` and absorbing of ``@``.

    Description
    -----------
    ``Zero @ A == A @ Zero == Zero``; dropped from sums on
    normalization. Its main consumer is the designed-for block
    layout (structural zeros). No computation, no halo. A singleton.
    """

    _instance: ClassVar[Zero | None] = None

    def __new__(cls) -> Zero:
        """Return the interned singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Domains unchanged (the zero map on the same signature)."""
        return domains[0] if len(domains) == 1 else domains

    def __call__(self, f: FieldLike) -> FieldLike:
        """Return the zero field on the operand's space."""
        return f.with_data(f.data * 0)


# ================================================================
#  Algebra objects: composition
# ================================================================
@final
class Composite(UnaryOperator):

    """
    Flat right-to-left factor chain; whole-space, not bindable.

    Description
    -----------
    Built by ``@`` (interned, D6); never nested (associativity).
    Mixed-axis or transform-bearing chains land here and inherit the
    raising ``__getitem__``. Halo **sums** along the chain (rules
    3.6); the symbol is the mode-wise product of factor symbols
    (rules 3.7). A binary-headed chain (Convolution) is designed-for.
    """

    def __init__(self, factors: tuple[Operator, ...]) -> None:
        """Plumbing constructor: build chains with ``@`` instead."""
        self._factors: tuple[Operator, ...] = tuple(factors)

    @property
    def factors(self) -> tuple[Operator, ...]:
        """The flat factor chain, outermost (applied last) first."""
        return self._factors

    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """Thread the (bare) domain through the chain right-to-left."""
        space = domain.bare
        for op in reversed(self._factors):
            space = resolve_codomain(op, space)
        return space

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """Per-axis SUM of factor halos along the chain (rules 3.6)."""
        return _chain_requirements(self._factors, domain)

    def eigenvalues(self, grid: object, space: SpaceLike) -> object:
        """Mode-wise product of factor symbols (rules 3.7)."""
        return _chain_eigenvalues(self._factors, grid, space)

    def _apply(self, f: FieldLike) -> FieldLike:
        """Apply the factors right-to-left (each application syncs)."""
        for op in reversed(self._factors):
            f = op(f)
        return f


@final
class SeparableComposite(SeparableOperator):

    """
    A chain of separable kernels on one axis; itself separable.

    Description
    -----------
    Produced by ``@`` when all factors are separable (or
    ``Dispatched`` holes) and axis-compatible (D5). Factors are
    stored **unbound** — the composite's ``bound_axis`` is the
    canonical binding, so ``(A @ B)["x"] is A["x"] @ B["x"]`` (D6).
    Overrides only the chain hooks: ``_apply_factor`` (run each
    factor on the axis, no intermediate sync — the summed halo covers
    the chain), ``requirements`` (sum), ``eigenvalues`` (product).
    """

    def __init__(self, factors: tuple[Operator, ...]) -> None:
        """Plumbing constructor: build chains with ``@`` instead.

        Factors are unbound separable kernels or ``Dispatched``
        holes.
        """
        self._factors: tuple[Operator, ...] = tuple(factors)

    @property
    def factors(self) -> tuple[Operator, ...]:
        """The unbound factor chain, outermost (applied last) first."""
        return self._factors

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Thread the 1D factor through the chain right-to-left."""
        space = domain
        for op in reversed(self._factors):
            space = op.codomain(space)
        return space

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """SUM of factor halos along the chain (rules 3.6)."""
        space = domain.bare
        if not isinstance(space, FunctionSpace):
            space = space.factor(_resolve_axis(self, space))
        return _chain_requirements(self._factors, space)

    def eigenvalues(self, grid: object, space: SpaceLike) -> object:
        """PRODUCT of factor symbols (rules 3.7)."""
        factors = self._factors
        if self.bound_axis is not None:
            # bind each factor to the composite's axis, like the
            # application path (``_apply_factor``): the stored factors
            # are unbound, but a symbol query on a multi-axis operand
            # must resolve the bound axis unambiguously.
            factors = tuple(op[self.bound_axis] for op in factors)
        return _chain_eigenvalues(factors, grid, space)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Run each factor's kernel on ``axis``, right-to-left.

        Description
        -----------
        Each stage is applied through its **bound** variant (D5:
        binding distributes over the chain), so kernels that resolve
        their codomain from ``self.bound_axis`` (the staggering
        family) stay unambiguous on multi-axis operands.
        """
        for op in reversed(self._factors):
            if isinstance(op, Dispatched):
                from fridom.framework2.grid.operators.registry import (  # noqa: PLC0415
                    DispatchError,
                )
                raise DispatchError(
                    f"unresolved Dispatched({op.kind!r}) in the "
                    "chain; resolve it against a registry (merge) "
                    "before application")
            f = op[axis]._apply_factor(f, axis)  # noqa: SLF001 — hook seam
        return f


# ================================================================
#  Algebra objects: sums and scalings
# ================================================================
@final
class OperatorSum(Operator):

    """
    Flat term list: ``(A + B)(f) == A(f) + B(f)``; same signature.

    Description
    -----------
    Built by ``+`` (interned); ``Zero`` terms drop on normalization.
    All terms must resolve the same codomain on a given domain
    (common-signature rule). Terms read the same input in parallel,
    so halo **maxes** (rules 3.6); the symbol is the sum of term
    symbols (rules 3.7).
    """

    def __init__(self, terms: tuple[Operator, ...]) -> None:
        """Plumbing constructor: build sums with ``+`` instead."""
        self._terms: tuple[Operator, ...] = tuple(terms)

    @property
    def terms(self) -> tuple[Operator, ...]:
        """The flat term list."""
        return self._terms

    def codomain(self, *domains: SpaceLike) -> SpaceLike:
        """Resolve the terms' common signature (or raise)."""
        if len(domains) != 1:
            raise NotImplementedError(
                "sums of non-unary operators are designed-for")
        codomains = [resolve_codomain(term, domains[0])
                     for term in self._terms]
        first = codomains[0]
        if any(c is not first for c in codomains):
            raise SpaceMismatchError(
                "the terms of an operator sum must share a "
                f"signature on {domains[0]!r}; resolved codomains "
                f"differ: {codomains!r}",
                left=domains[0], operation="+")
        return first

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """Per-axis MAX over the terms (rules 3.6)."""
        halo = 0
        layout: Literal["any", "local", "transpose"] = "any"
        collective = False
        for term in self._terms:
            req = term.requirements(domain)
            halo = max(halo, req.halo)
            layout = _merge_layout(layout, req.layout)
            collective = collective or req.collective
        return OperatorRequirements(
            halo=halo, layout=layout, collective=collective)

    def eigenvalues(self, grid: object, space: SpaceLike) -> object:
        """SUM of the term symbols (rules 3.7)."""
        total = None
        for term in self._terms:
            sym = term.eigenvalues(grid, space)
            total = sym if total is None else total + sym
        return total

    def __call__(self, f: FieldLike) -> FieldLike:
        """Validate the common signature, then apply and add."""
        self.codomain(f.function_space.bare)
        return self._apply(f)

    def _apply(self, f: FieldLike) -> FieldLike:
        """Apply every term to ``f`` and add the (synced) results."""
        results = [term(f) for term in self._terms]
        total = results[0]
        for result in results[1:]:
            total = total + result
        return total


@final
@partial(fr.utils.jaxify, dynamic=("coeff",))
class ScaledOperator(Operator):

    """
    Coefficient-scaled operator: ``(c * A)(f) == A(f) * c``.

    Description
    -----------
    The rules-3.4 machinery: a field coefficient multiplies the
    *output* (so ``c`` lives on ``A``'s codomain). The coefficient is
    the cluster's dynamic pytree leaf; the algebraic structure is
    static. A genuine field coefficient breaks translation
    invariance, so the scaled operator has a symbol only for constant
    ``c``. Not interned: the coefficient is a value, not structure
    (see the class-doc deviation note in the Wave-2 report).
    """

    def __init__(
        self, coeff: complex | FieldLike, target: Operator,
    ) -> None:
        """Store the coefficient (dynamic leaf) and the target."""
        #: complex scalar, or a field on ``target``'s codomain
        self.coeff: complex | FieldLike = coeff
        #: the scaled operator (static structure)
        self.target: Operator = target

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Resolve the target's codomain (the scaling is pointwise)."""
        if len(domains) == 1:
            return resolve_codomain(self.target, domains[0])
        return self.target.codomain(*domains)

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """Return the target's requirements (pointwise product)."""
        return self.target.requirements(domain)

    def eigenvalues(self, grid: object, space: SpaceLike) -> object:
        """Scale the target's symbol iff the coefficient is a scalar.

        Description
        -----------
        A scalar coefficient is translation-invariant, so the scaled
        operator keeps a symbol: a Python number, or a 0-d ``jax.Array``
        (a traced-but-constant scalar, e.g. the ``1/dsqr`` leaf riding
        ``ctx.params``). A genuine **field** coefficient (a
        ``ScalarField`` / anything with ``function_space``, or an n-d
        array) breaks translation invariance and has no symbol.
        """
        coeff = self.coeff
        if isinstance(coeff, int | float | complex) or (
                isinstance(coeff, jax.Array) and coeff.ndim == 0):
            return coeff * self.target.eigenvalues(grid, space)
        raise EigenbasisError(
            "a field coefficient breaks translation invariance; the "
            "scaled operator has no symbol")

    def __call__(self, f: FieldLike) -> FieldLike:
        """Apply the target, then scale the (synced) output."""
        return self._apply(f)

    def _apply(self, f: FieldLike) -> FieldLike:
        """``A(f) * c`` through the field's arithmetic."""
        return self.target(f) * self.coeff


# ================================================================
#  Algebra objects: blocks (designed-for) and placeholders
# ================================================================
@final
class Block(Operator):

    """
    Block matrix of scalar-signature operators (designed-for, B1).

    Description
    -----------
    Lands with ``grad``/``div``/``curl``/``laplacian`` (D7); the
    class is declared so the module surface matches the class doc.
    """

    def __init__(self, blocks: object) -> None:
        """Designed-for: raise until the block layout lands."""
        raise NotImplementedError(
            "Block is designed-for (D7); it lands with the "
            "grad/div/curl/laplacian builders")

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Tuple codomain (designed-for)."""
        raise NotImplementedError(
            "Block is designed-for (D7)")


@final
class Dispatched(Operator):

    """
    A registry kind placeholder, resolved when the grid is known.

    Description
    -----------
    One object, two roles (D4): a chain factor inside a registered
    default (resolved at merge/assembly, baked concrete) and the
    user-facing verb (resolved at application against
    ``f.grid.dispatch``). The constructor is a public extension
    escape-hatch (D3b); an unknown kind is a clean ``DispatchError``
    at resolution. Instances are interned on ``(kind, axis)``.
    """

    def __new__(cls, kind: str) -> Dispatched:
        """Return the interned placeholder for ``kind``."""
        if not isinstance(kind, str):
            raise TypeError(
                f"dispatch kinds are strings, got {kind!r}")

        def build() -> Dispatched:
            obj = super(Dispatched, cls).__new__(cls)
            obj._kind = kind
            obj._axis = None
            return obj

        return _ALGEBRA_TABLE.intern((cls, kind, None), build)

    def __init__(self, kind: str) -> None:
        """No-op: attributes are set in the interning ``__new__``."""

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def kind(self) -> str:
        """The registry kind this placeholder resolves."""
        return self._kind

    @property
    def bound_axis(self) -> str | None:
        """Pending axis bind carried onto the resolved operator."""
        return self._axis

    @property
    def unbound(self) -> Dispatched:
        """The pending-bind-free placeholder for the same kind."""
        return Dispatched(self._kind)

    # ------------------------------------------------------------
    #  Binding and resolution
    # ------------------------------------------------------------
    def __getitem__(self, axis: str) -> Dispatched:
        """
        Carry a pending axis bind onto the (later) resolved operator.

        Parameters
        ----------
        axis : str
            The coordinate name to bind after resolution.

        Returns
        -------
        Dispatched
            The interned placeholder variant with the pending bind.
        """
        if not isinstance(axis, str):
            raise TypeError(
                f"axes are bound by coordinate name, got {axis!r}")
        if self._axis is not None:
            raise TypeError(
                f"placeholder is already bound to {self._axis!r}; "
                "rebinding is not allowed")
        kind = self._kind

        def build() -> Dispatched:
            obj = object.__new__(Dispatched)
            obj._kind = kind  # noqa: SLF001 — interning factory
            obj._axis = axis  # noqa: SLF001 — interning factory
            return obj

        return _ALGEBRA_TABLE.intern(("bound", self, axis), build)

    def resolve(
        self, registry: OperatorRegistry, space: SpaceLike,
    ) -> Operator:
        """
        Replace the placeholder with the registered operator.

        Parameters
        ----------
        registry : OperatorRegistry
            The registry to resolve against.
        space : SpaceLike
            The (factor) space keying the resolution.

        Returns
        -------
        Operator
            The registered operator, with any pending axis bound.
        """
        op = registry.resolve(self._kind, space)
        if self._axis is not None:
            op = op[self._axis]
        return op

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Standalone verb: resolve against ``f.grid.dispatch``, apply.

        Parameters
        ----------
        f : FieldLike
            The operand field; supplies the grid and the factor
            space keying the resolution.

        Returns
        -------
        FieldLike
            The resolved operator's result.
        """
        space = f.function_space.bare
        if self._axis is not None:
            factor = space.factor(self._axis)
        else:
            axes = _bindable_axes(space)
            if len(axes) != 1:
                raise ValueError(
                    f"cannot pick an axis for {self._kind!r} on "
                    f"{space!r}; bind explicitly: "
                    f"Dispatched({self._kind!r})[axis]")
            factor = space.factor(axes[0])
        return self.resolve(f.grid.dispatch, factor)(f)

    def codomain(
        self,
        *domains: SpaceLike,  # noqa: ARG002 — always raises
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Unresolved placeholders have no signature: raise."""
        from fridom.framework2.grid.operators.registry import (  # noqa: PLC0415
            DispatchError,
        )
        raise DispatchError(
            f"Dispatched({self._kind!r}) is an unresolved "
            "placeholder; resolve it against a registry before "
            "querying its signature")


# ================================================================
#  Shared resolution helpers
# ================================================================
def resolve_codomain(op: Operator, domain: SpaceLike) -> SpaceLike:
    """
    Resolve the bare codomain of ``op`` applied on ``domain``.

    Description
    -----------
    The shared application-path resolver: strips the layout, lifts a
    separable kernel's per-factor ``codomain`` over the (bound or
    sole-factor) product axis (identity along ``ConstantSpace``
    factors), and defers to ``op.codomain`` for whole-space
    operators. Layout re-attachment is the caller's job.

    Parameters
    ----------
    op : Operator
        The operator to resolve.
    domain : SpaceLike
        The operand's (possibly laid-out) space.

    Returns
    -------
    SpaceLike
        The bare codomain space.
    """
    bare = domain.bare
    if isinstance(op, SeparableOperator):
        axis = _resolve_axis(op, bare)
        factor = bare.factor(axis)
        if isinstance(factor, ConstantSpace):
            return bare
        new_factor = op.codomain(factor)
        if isinstance(bare, FunctionSpace):
            return new_factor
        return bare.replace(**{axis: new_factor})
    return op.codomain(bare)


def _bindable_axes(space: SpaceLike) -> tuple[str, ...]:
    """Coordinate names of the non-``ConstantSpace`` factors."""
    return space.active_axis_names


def _resolve_axis(op: SeparableOperator, space: SpaceLike) -> str:
    """
    Resolve the axis a separable kernel acts on for ``space``.

    Description
    -----------
    ``bound_axis`` if set (validated against the space's names), else
    the space's sole bindable factor, else raise asking for an
    explicit bind (D2).

    Parameters
    ----------
    op : SeparableOperator
        The (possibly bound) separable kernel.
    space : SpaceLike
        The (bare) operand space.

    Returns
    -------
    str
        The resolved coordinate name.
    """
    axis = op.bound_axis
    if axis is not None:
        if axis not in space.names:
            raise SpaceMismatchError(
                f"operator is bound to axis {axis!r}, but the "
                f"operand's names are {space.names}",
                left=space, operation="__call__")
        return axis
    candidates = _bindable_axes(space)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"no bindable factor on {space!r}; every factor is a "
            "ConstantSpace")
    raise ValueError(
        f"ambiguous axis on {space!r}: bind explicitly (op[axis]) "
        f"among {candidates}")


def _finalize(
    operand: FieldLike, result: FieldLike, codomain: SpaceLike,
) -> FieldLike:
    """
    Re-attach the domain layout to the kernel result.

    Description
    -----------
    The tail of the shared application path: the kernel built
    ``result`` on the bare codomain; this re-attaches the operand's
    layout through the field's plumbing constructor (the kernel's
    halo-validity claim is kept). A codomain that already carries a
    layout is a layout-transition operator (``Reshard``) and is kept
    as resolved. Under the consumption-side sync contract (task 1.8)
    no sync is appended here: the operand was made valid *before*
    the kernel ran (``_ensure_valid``), and the result's ghost
    validity is whatever its construction seam certified.

    Parameters
    ----------
    operand : FieldLike
        The (outermost) operand field.
    result : FieldLike
        The kernel result field.
    codomain : SpaceLike
        The resolved codomain (bare, unless layout-transitioning).

    Returns
    -------
    FieldLike
        The result on the laid-out codomain.
    """
    layout = operand.function_space.layout
    if codomain.layout is not None:
        expected = codomain  # layout transition (Reshard)
    elif layout is not None:
        expected = codomain.with_layout(layout)
    else:
        expected = codomain
    space = result.function_space
    if space is not expected:
        if space.bare is not expected.bare:
            raise SpaceMismatchError(
                f"kernel returned a field on {space!r}, but the "
                f"resolved codomain is {expected!r}",
                left=space, right=expected, operation="__call__")
        result = type(result)(
            result.grid, expected,
            result._data,  # noqa: SLF001 — plumbing-constructor seam
            result.metadata,
            halo_valid=result.halo_valid)
    return result


def _required_halo(op: Operator, space: SpaceLike) -> dict[str, int]:
    """
    Per-name ghost depth one application of ``op`` consumes.

    Description
    -----------
    The runtime twin of ``HaloTracer._grown`` (task 1.8): per-axis
    for separable kernels (their sole applied factor), every
    bindable axis for whole-space operators (conservative).

    Parameters
    ----------
    op : Operator
        The operator being applied.
    space : SpaceLike
        The operand's (laid-out) function space.

    Returns
    -------
    dict[str, int]
        Positive per-name depths; empty for halo-0 applications.
    """
    bare = space.bare
    if isinstance(op, SeparableOperator):
        axis = _resolve_axis(op, bare)
        factor = (bare if not hasattr(bare, "factor")
                  else bare.factor(axis))
        if isinstance(factor, ConstantSpace):
            return {}
        halo = op.requirements(factor).halo
        return {axis: halo} if halo else {}
    halo = op.requirements(bare).halo
    if not halo:
        return {}
    return {name: halo
            for factor in bare.factors
            if not isinstance(factor, ConstantSpace)
            for name in factor.names}


def _ensure_valid(
    f: FieldLike, required: dict[str, int],
) -> FieldLike:
    """
    Sync the operand iff its ghost validity is below ``required``.

    Description
    -----------
    The consumption-side sync placement (task 1.8): a trace-time
    check of static Python attributes — zero runtime cost under jit.
    A triggered sync fills every axis to the negotiated widths, and
    the synced field is memoized in an external identity-keyed cache
    (:data:`_SYNC_CACHE`) — never written back onto the operand, whose
    ``halo_valid`` is treedef-participating static aux — so further
    consumers of the same field reuse it: n readers pay one exchange.

    Parameters
    ----------
    f : FieldLike
        The operand field.
    required : dict[str, int]
        Per-name depths this application consumes.

    Returns
    -------
    FieldLike
        ``f`` itself when already valid, else the synced field.
    """
    if not required:
        return f
    valid = dict(f.halo_valid.widths)
    if all(valid.get(name, 0) >= depth
           for name, depth in required.items()):
        return f
    cached = _sync_cache_get(f)
    if cached is not None:
        cvalid = dict(cached.halo_valid.widths)
        if all(cvalid.get(name, 0) >= depth
               for name, depth in required.items()):
            return cached
    synced = _sync_node()(f)
    if synced is not f:
        _memoize_sync(f, synced)
    return synced


#: identity-keyed synced-ghost cache: the memoized exchange lives
#: here, NOT on the operand's own attributes, so a carry-resident
#: field's treedef (``halo_valid`` is treedef-participating static
#: aux) is never mutated. WeakKeyDictionary auto-evicts per-trace
#: field objects once the trace ends.
_SYNC_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _sync_cache_get(f: FieldLike) -> FieldLike | None:
    """Return a previously-memoized synced view of ``f``, if any."""
    try:
        return _SYNC_CACHE.get(f)
    except TypeError:
        return None


def _memoize_sync(f: FieldLike, synced: FieldLike) -> None:
    """
    Store the synced view in the identity-keyed cache (no mutation).

    Description
    -----------
    The memoized exchange is recorded in an external identity-keyed
    cache instead of being written back onto ``f``'s own ``_data`` /
    ``_halo_valid`` attributes. Repeated consumers of the same object
    still pay one exchange (they hit the cache), but the operand's
    pytree treedef (``halo_valid`` is treedef-participating static
    aux) is never mutated, so a carry-resident field passed unchanged
    through ``lax.scan`` keeps a stable treedef. Guard: a *concrete*
    field consumed inside someone else's trace must not cache a
    tracer.
    """
    old = f._data  # noqa: SLF001 — documented storage seam
    new = synced._data  # noqa: SLF001 — documented storage seam
    if (not isinstance(old, jax.core.Tracer)
            and isinstance(new, jax.core.Tracer)):
        return
    with contextlib.suppress(TypeError):
        _SYNC_CACHE[f] = synced


def _sync_node() -> Operator:
    """Return the ``Sync`` singleton (lazy: movement imports base)."""
    from fridom.framework2.grid.operators.movement import (  # noqa: PLC0415 — import cycle seam
        Sync,
    )
    return Sync()


def _merge_layout(
    a: Literal["any", "local", "transpose"],
    b: Literal["any", "local", "transpose"],
) -> Literal["any", "local", "transpose"]:
    """Combine two layout demands; conflicting demands raise."""
    if b in {a, "any"}:
        return a
    if a == "any":
        return b
    raise ValueError(
        f"conflicting layout requirements: {a!r} vs {b!r}")


def _chain_requirements(
    factors: tuple[Operator, ...], domain: SpaceLike,
) -> OperatorRequirements:
    """Accumulate requirements along a chain (halo sums, rules 3.6)."""
    space = domain.bare
    halo = 0
    layout: Literal["any", "local", "transpose"] = "any"
    collective = False
    for op in reversed(factors):
        req = op.requirements(space)
        halo += req.halo
        layout = _merge_layout(layout, req.layout)
        collective = collective or req.collective
        space = resolve_codomain(op, space)
    return OperatorRequirements(
        halo=halo, layout=layout, collective=collective)


def _chain_eigenvalues(
    factors: tuple[Operator, ...], grid: object, space: SpaceLike,
) -> object:
    """Mode-wise product of factor symbols along a chain (rules 3.7)."""
    total = None
    current = space
    for op in reversed(factors):
        sym = op.eigenvalues(grid, current)
        total = sym if total is None else sym @ total
        current = resolve_codomain(op, current)
    return total


# ================================================================
#  Normalization and interning of the algebra (D5/D6)
# ================================================================
def _chain_factors(op: Operator) -> tuple[Operator, ...]:
    """Flatten ``op`` into chain factors (bound composites re-bind)."""
    if isinstance(op, SeparableComposite):
        if op.bound_axis is None:
            return op.factors
        return tuple(factor[op.bound_axis] for factor in op.factors)
    if isinstance(op, Composite):
        return op.factors
    return (op,)


def _separable_axis(
    factors: tuple[Operator, ...],
) -> tuple[bool, str | None]:
    """
    Decide separable-compatibility of a flat chain.

    Description
    -----------
    A chain is separable iff every factor is a separable kernel (or
    a ``Dispatched`` hole) and the bound axes are compatible: all
    unbound, or all bound to the same axis (D5).

    Parameters
    ----------
    factors : tuple[Operator, ...]
        The flat chain factors.

    Returns
    -------
    tuple[bool, str | None]
        ``(separable, common_axis)``; the axis is None when unbound.
    """
    axes = set()
    for op in factors:
        if not isinstance(op, SeparableOperator | Dispatched):
            return False, None
        axes.add(op.bound_axis)
    if len(axes) != 1:
        return False, None
    return True, axes.pop()


def _compose(left: Operator, right: Operator) -> Operator:
    """Flatten, normalize, and intern the chain ``left @ right``."""
    factors = (*_chain_factors(left), *_chain_factors(right))
    if any(isinstance(op, Zero) for op in factors):
        return Zero()
    factors = tuple(
        op for op in factors if not isinstance(op, Identity))
    if not factors:
        return Identity()
    if len(factors) == 1:
        return factors[0]
    separable, axis = _separable_axis(factors)
    if separable:
        unbound = tuple(op.unbound for op in factors)
        chain = _ALGEBRA_TABLE.intern(
            ("sep", unbound), lambda: SeparableComposite(unbound))
        return chain if axis is None else chain[axis]
    return _ALGEBRA_TABLE.intern(
        ("chain", factors), lambda: Composite(factors))


def _sum_terms(op: Operator) -> tuple[Operator, ...]:
    """Flatten ``op`` into sum terms."""
    if isinstance(op, OperatorSum):
        return op.terms
    return (op,)


def _sum(left: Operator, right: Operator) -> Operator:
    """Flatten, normalize, and intern the sum ``left + right``."""
    terms = (*_sum_terms(left), *_sum_terms(right))
    terms = tuple(t for t in terms if not isinstance(t, Zero))
    if not terms:
        return Zero()
    if len(terms) == 1:
        return terms[0]
    return _ALGEBRA_TABLE.intern(
        ("sum", terms), lambda: OperatorSum(terms))
