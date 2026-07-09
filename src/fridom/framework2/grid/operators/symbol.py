r"""
``Symbol``: a diagonal operator on a coefficient space.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_base.md``
(section "Symbol") and ``notes/framework2/operator_symbols_plan.md``
(sections 3-4). The eigenvalue / filter / mask primitive: a thin
wrapper over one materialized per-mode diagonal array, carrying the
``(space, codomain)`` coefficient tags as static pytree aux and the
diagonal ``_data`` as the single dynamic leaf.

A ``Symbol`` is **not** a ``ScalarField``: its ``*``/``+``/``**``/
``1 / .`` are the diagonal-operator algebra (composition, inverse),
never the physical product, and broadcasting across product factors
is diagonal-operator extension (``Identity ⊗ D``, plain jax size-1
broadcasting) rather than the field constant-lift. It implements the
unary-apply protocol (``__call__``, ``space``/``codomain``) so
solvers can treat it as a diagonal map.
"""
# Wave 9A: Symbol
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, final

import jax.numpy as jnp

import fridom.framework as fr
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.operators.transform import axis_vector
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.composition import (
    compose_spaces,
    union_spaces,
)
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.function_space import FunctionSpace
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.operators.base import FieldLike
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


@final
@partial(fr.utils.jaxify, dynamic=("_data",))
class Symbol:

    """
    Diagonal operator on a coefficient space (eigenvalues, filters).

    Description
    -----------
    Wraps the per-mode diagonal ``data`` as a diagonal operator from
    ``space`` to ``codomain`` (equal unless the symbol retags, e.g. a
    staggering first-derivative ``Fourier(Center) -> Fourier(Right)``).
    The tags are interned coefficient spaces (static aux); ``data`` is
    the dynamic pytree leaf, pre-shaped for size-1 broadcasting across
    the product's other (``ConstantSpace``) factors.

    Parameters
    ----------
    space : SpaceLike
        Domain coefficient space of the diagonal (stored bare).
    data : jax.Array
        The per-mode diagonal values, broadcast-shaped over ``space``.
    codomain : SpaceLike | None, optional
        Codomain space; ``None`` reuses ``space`` (default: None).
    """

    def __init__(
        self,
        space: SpaceLike,
        data: jax.Array,
        codomain: SpaceLike | None = None,
    ) -> None:
        """Wrap per-mode values ``data`` as a diagonal on ``space``."""
        self._space: SpaceLike = space.bare
        self._codomain: SpaceLike = (
            self._space if codomain is None else codomain.bare)
        self._data: jax.Array = jnp.asarray(data)

    @classmethod
    def from_field(cls, f: FieldLike) -> Symbol:
        """
        Build a symbol from a coordinate ``ScalarField``.

        Description
        -----------
        The sanctioned crossing from the coefficient-coordinate
        accessors (``grid.wavenumbers`` / ``grid.measure``): the
        symbol adopts the field's array as its diagonal and the
        field's (bare) space as both tags.

        Parameters
        ----------
        f : FieldLike
            A coordinate field (e.g. ``grid.wavenumbers(space)``).

        Returns
        -------
        Symbol
            The diagonal wrapping ``f.data`` on ``f``'s bare space.
        """
        return cls(f.function_space.bare, f.data)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def space(self) -> SpaceLike:
        """Domain coefficient space of the diagonal."""
        return self._space

    @property
    def domain(self) -> SpaceLike:
        """Domain coefficient space (``RealizedMap`` alias of ``space``)."""
        return self._space

    @property
    def codomain(self) -> SpaceLike:
        """Codomain space (equals ``space`` unless retagging)."""
        return self._codomain

    @property
    def data(self) -> jax.Array:
        """The per-mode diagonal values (dynamic leaf)."""
        return self._data

    # ================================================================
    #  Adjoint and (pseudo-)inverse
    # ================================================================
    def conj(self) -> Symbol:
        """
        Complex-conjugate symbol (adjoint of the diagonal).

        Description
        -----------
        The Hermitian adjoint of a diagonal map conjugates the
        entries and swaps the domain/codomain tags (a no-op on the
        tags for the common non-retagging symbol).

        Returns
        -------
        Symbol
            The adjoint diagonal.
        """
        return Symbol(self._codomain, jnp.conj(self._data),
                      codomain=self._space)

    def inverse(self, where_zero: complex = 0.0) -> Symbol:
        r"""
        Pseudo-inverse; zero diagonal entries become ``where_zero``.

        Description
        -----------
        The spectral-solve primitive: every **structural** zero of
        the diagonal — the Poisson ``k = 0`` nullspace and the
        real-FFT Nyquist-zeroed modes, both set exactly by the
        operators — maps to ``where_zero`` via an exact ``== 0`` test
        (no floating tolerance), and the tags flip. So
        ``lap.inverse()(-div_hat)`` regularizes ``k = 0`` without
        caller-side masking.

        Parameters
        ----------
        where_zero : complex, optional
            The inverse value at structural zeros (default: 0.0).

        Returns
        -------
        Symbol
            The (tag-flipped) pseudo-inverse diagonal.
        """
        zero = self._data == 0
        safe = jnp.where(zero, jnp.ones_like(self._data), self._data)
        inv = jnp.where(zero, where_zero, 1.0 / safe)
        return Symbol(self._codomain, inv, codomain=self._space)

    # ================================================================
    #  Magnitude and elementwise root
    # ================================================================
    @property
    def magnitude(self) -> Symbol:
        r"""
        The real ``|lambda|`` diagonal, collapsed to the domain tags.

        Description
        -----------
        The honest magnitude of the diagonal:
        ``sqrt(real(conj(d) * d))``, retagged **collapsed** onto
        ``(domain, domain)`` (``codomain = space``), so a retagging
        symbol's magnitude is endo. ``D.magnitude ** 2`` is the
        ``|k_hat|**2`` / ``|o_hat|**2`` dispersion quantity, equal to
        ``(D.conj() @ D).data.real``. The magnitude of a
        backward-threaded symbol is face-side-endo, of a forward one
        centre-side-endo — same data, different tags. Structural
        zeros stay exact (``sqrt(0) == 0``).

        Returns
        -------
        Symbol
            The real magnitude diagonal on ``(space, space)``.
        """
        return Symbol(self._space, jnp.sqrt(
            jnp.real(jnp.conj(self._data) * self._data)))

    def sqrt(self) -> Symbol:
        """
        Elementwise square root; forbidden across a retag.

        Description
        -----------
        The diagonal root (``sqrt(d)``, bare jax branch-cut
        semantics), tags kept. Like ``**`` it is only meaningful on
        an endo diagonal — a retagging symbol has no elementwise
        root within one tag pair.

        Returns
        -------
        Symbol
            The rooted diagonal (``codomain is space`` required).
        """
        if self._codomain is not self._space:
            raise SpaceMismatchError(
                "Symbol.sqrt() needs codomain is space (collapse a "
                "retagging symbol via .magnitude first)",
                left=self._space, right=self._codomain,
                operation="Symbol.sqrt")
        return Symbol(self._space, jnp.sqrt(self._data),
                      codomain=self._codomain)

    # ================================================================
    #  Application (Hadamard multiply)
    # ================================================================
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply to a field: elementwise (Hadamard) multiply.

        Description
        -----------
        Multiplies on the strict same-space precondition
        (``f.function_space.bare is space``) and emits on ``codomain``
        (through the field plumbing constructor when the symbol
        retags).

        Parameters
        ----------
        f : FieldLike
            The operand field, on the symbol's ``space``.

        Returns
        -------
        FieldLike
            The Hadamard product, on the symbol's ``codomain``.
        """
        laid_out = f.function_space
        if laid_out.bare is not self._space:
            raise SpaceMismatchError(
                f"symbol on {self._space!r} applied to a field on "
                f"{laid_out.bare!r}", left=laid_out.bare,
                right=self._space, operation="Symbol.__call__")
        out = f.data * self._data
        if self._codomain is self._space:
            return f.with_data(out)
        layout = laid_out.layout
        codomain = (self._codomain.with_layout(layout)
                    if layout is not None else self._codomain)
        stored = store(f.grid.decomposition, codomain,
                       jnp.asarray(out))
        return type(f)(f.grid, codomain, stored, f.metadata)

    # ================================================================
    #  Diagonal (elementwise) algebra — never the physical product
    # ================================================================
    def __mul__(self, other: Symbol | complex | FieldLike) -> Symbol:
        """Elementwise product (a symbol, a scalar, or a field)."""
        if _is_field(other):
            return self._mul_field(other)
        return self._elementwise(other, jnp.multiply)

    def __rmul__(self, other: complex | FieldLike) -> Symbol | object:
        """Scalar/field product (commutative in the coefficient)."""
        if _is_field(other):
            return self._mul_field(other)
        if _is_scalar(other):
            return Symbol(self._space, self._data * other,
                          codomain=self._codomain)
        return NotImplemented

    def __add__(self, other: Symbol | complex) -> Symbol:
        """Elementwise sum (the Laplacian ``kx**2 + ky**2`` union)."""
        return self._elementwise(other, jnp.add)

    def __radd__(self, other: complex) -> Symbol | object:
        """Scalar sum (commutative)."""
        if _is_scalar(other):
            return Symbol(self._space, self._data + other,
                          codomain=self._codomain)
        return NotImplemented

    def __sub__(self, other: Symbol | complex) -> Symbol:
        """Elementwise difference."""
        return self._elementwise(other, jnp.subtract)

    def __neg__(self) -> Symbol:
        """Negation of the diagonal."""
        return Symbol(self._space, -self._data,
                      codomain=self._codomain)

    def __pow__(self, p: int) -> Symbol:
        """
        Elementwise power; forbidden across a retag.

        Parameters
        ----------
        p : int
            The exponent.

        Returns
        -------
        Symbol
            The powered diagonal (``codomain is space`` required).
        """
        if self._codomain is not self._space:
            raise SpaceMismatchError(
                "Symbol ** p needs codomain is space (a retagging "
                "symbol squares via bwd @ fwd, not ** 2)",
                left=self._space, right=self._codomain,
                operation="Symbol.__pow__")
        return Symbol(self._space, self._data ** p,
                      codomain=self._codomain)

    def __truediv__(self, other: Symbol | complex) -> Symbol | object:
        """Elementwise division (bare jax semantics at zeros)."""
        if isinstance(other, Symbol):
            return self._elementwise(other, jnp.divide)
        if _is_scalar(other):
            return Symbol(self._space, self._data / other,
                          codomain=self._codomain)
        return NotImplemented

    def __rtruediv__(self, other: complex) -> Symbol | object:
        """Scalar over the diagonal: the (tag-flipped) raw inverse."""
        if _is_scalar(other):
            return Symbol(self._codomain, other / self._data,
                          codomain=self._space)
        return NotImplemented

    def __matmul__(self, other: object) -> Symbol | object:
        """
        Diagonal composition ``(A @ B)(f) == A(B(f))``.

        Description
        -----------
        A ``Symbol`` inner is fused **eagerly** (the bitwise Hadamard
        fast path, unchanged): factor-wise (the ``Identity ⊗ D``
        tensor-product extension) per axis the shared physical mode
        index is ``B.codomain`` met with ``A.space`` — required
        identical where **both** are non-``Constant``,
        disjoint/passthrough where either is ``Constant``. The composed
        domain reads ``B.space`` (else ``A.space`` where ``B`` is
        identity along the axis), the composed codomain reads
        ``A.codomain`` (else ``B.codomain`` where ``A`` is identity),
        and the leaves multiply (jax size-1 broadcasting builds the
        tensor product across disjoint axes). On the common same-space
        chain (``B.codomain is A.space``) this reduces to
        ``space = B.space``, ``codomain = A.codomain`` — the honest
        discrete Laplacian ``bwd @ fwd``.

        Any other :class:`RealizedMap` inner (a bound transform, a
        realized composite) builds a lazy ``RealizedComposite`` (fusing
        adjacent symbols); an *unmaterialized* :class:`Operator` raises
        the taught materialization guard.

        Parameters
        ----------
        other : object
            The inner map (applied first): a ``Symbol``, another
            realized map, or (rejected) an operator recipe.

        Returns
        -------
        Symbol | object
            The fused ``Symbol``, a ``RealizedComposite``, or
            ``NotImplemented``.
        """
        if isinstance(other, Symbol):
            if other._codomain is self._space:
                return Symbol(other._space, self._data * other._data,
                              codomain=self._codomain)
            space, codomain = compose_spaces(
                other._space, other._codomain,
                self._space, self._codomain)
            return Symbol(space, self._data * other._data,
                          codomain=codomain)
        from fridom.framework2.grid.operators.realized import (  # noqa: PLC0415
            realized_matmul,
        )
        return realized_matmul(self, other)

    def __rmatmul__(self, other: object) -> object:
        """
        Reflected composition ``other @ self``.

        Description
        -----------
        Reached when ``other`` did not implement ``@`` for a
        ``Symbol`` — e.g. an unmaterialized :class:`Operator` recipe,
        which raises the taught materialization guard. ``Symbol @
        Symbol`` and ``realized @ Symbol`` never route here (the left
        operand handles them).

        Parameters
        ----------
        other : object
            The outer (applied-last) operand.

        Returns
        -------
        object
            ``NotImplemented`` unless the guard raises.
        """
        from fridom.framework2.grid.operators.realized import (  # noqa: PLC0415
            realized_rmatmul,
        )
        return realized_rmatmul(self, other)

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _elementwise(
        self, other: Symbol | complex, op: object,
    ) -> Symbol:
        """
        Combine two symbols elementwise on the tag-union spaces.

        Description
        -----------
        The ``+`` / ``*`` / ``-`` / ``/`` path: a scalar combines
        into the diagonal (tags kept); a symbol is combined on the
        factor-wise tag union (``Constant ⊗ X -> X``) after size-1
        broadcasting of both diagonals.
        """
        if isinstance(other, Symbol):
            operation = getattr(op, "__name__", "op")
            space = union_spaces(
                self._space, other.space, operation=operation)
            codomain = union_spaces(
                self._codomain, other.codomain, operation=operation)
            return Symbol(space, op(self._data, other.data),
                          codomain=codomain)
        if _is_scalar(other):
            return Symbol(self._space, op(self._data, other),
                          codomain=self._codomain)
        return NotImplemented

    def _mul_field(self, field: FieldLike) -> Symbol:
        r"""
        Multiply the diagonal by a coefficient ``ScalarField``.

        Description
        -----------
        The ``Symbol x field`` coefficient rule (symbol_stack_design.md
        §"``Symbol x field``"): because multiplication by a coefficient
        constant in ``x`` commutes with ``FFT_x``
        (``kx · FFT_x(c(y) f) = kx · c(y) · FFT_x(f)``), the product is
        legal **iff** ``field`` is ``ConstantSpace`` on every
        *transformed* (coefficient) factor of the symbol; it may vary on
        the symbol's ``Constant``/nodal (physical) factors. The result
        adopts the field's space there (``Constant(y) -> Nodal(y)``) and
        Hadamard-multiplies the data (jax size-1 broadcasting), giving
        the mixed representation (Fourier in ``x``, physical in ``y``).
        The degenerate all-``Constant`` field is the ``dsqr`` scalar; a
        genuine profile ``c(y)`` / ``N^2(z)`` is the general case — one
        code path.

        Parameters
        ----------
        field : FieldLike
            The coefficient field, constant on the transformed factors.

        Returns
        -------
        Symbol
            The diagonal times the coefficient, in the mixed space.
        """
        fspace = field.function_space.bare
        s_domain = self._space.factors
        s_codomain = self._codomain.factors
        f_factors = fspace.factors
        if not len(s_domain) == len(s_codomain) == len(f_factors):
            raise SpaceMismatchError(
                f"cannot multiply a symbol on {self._space!r} by a "
                f"field on {fspace!r}: incompatible ranks",
                left=self._space, right=fspace, operation="Symbol.__mul__")
        domain, codomain = [], []
        for sd, sc, ff in zip(s_domain, s_codomain, f_factors,
                              strict=True):
            if isinstance(sd, CoefficientSpace):
                if not isinstance(ff, ConstantSpace):
                    raise SpaceMismatchError(
                        "a Symbol x field coefficient must be constant "
                        f"on every transformed factor; {ff!r} varies on "
                        f"the transformed axis {sd!r}", left=sd,
                        right=ff, operation="Symbol.__mul__")
                domain.append(sd)
                codomain.append(sc)
            else:
                domain.append(_adopt(sd, ff))
                codomain.append(_adopt(sc, ff))
        return Symbol(TensorProductSpace.of(*domain),
                      self._data * field.data,
                      codomain=TensorProductSpace.of(*codomain))


# ================================================================
#  Diagonal-symbol construction helpers
# ================================================================
def _is_field(obj: object) -> bool:
    """Whether ``obj`` is a coefficient field (not a Symbol/scalar)."""
    return hasattr(obj, "function_space") and hasattr(obj, "data")


def _is_scalar(obj: object) -> bool:
    """Whether ``obj`` is a scalar coefficient of the diagonal.

    Description
    -----------
    A Python number or a 0-d array (e.g. a traced ``ctx.params`` leaf
    such as ``1/dsqr``). An n-d array is **not** a scalar — it carries
    no space tags, so it must enter as a ``ScalarField`` (via
    ``_mul_field``) where the constant-on-transformed-axes guard applies.
    """
    if isinstance(obj, int | float | complex):
        return True
    return not _is_field(obj) and getattr(obj, "ndim", None) == 0


def _adopt(sym_factor: SpaceLike, field_factor: SpaceLike) -> SpaceLike:
    """Union a non-transformed symbol factor with the field's factor."""
    if isinstance(sym_factor, ConstantSpace):
        return field_factor
    if isinstance(field_factor, ConstantSpace):
        return sym_factor
    if sym_factor is field_factor:
        return sym_factor
    raise SpaceMismatchError(
        "a Symbol x field coefficient disagrees with the symbol's own "
        f"physical factor: {sym_factor!r} vs {field_factor!r}",
        left=sym_factor, right=field_factor, operation="Symbol.__mul__")


def diagonal_symbol(
    bare: SpaceLike,
    axis: str,
    domain_factor: SpaceLike,
    codomain_factor: SpaceLike,
    leaf: jax.Array,
) -> Symbol:
    """
    Lift a 1D per-mode ``leaf`` into a diagonal ``Symbol``.

    Description
    -----------
    The shared body of the per-operator ``eigenvalues`` methods:
    tags the diagonal with the coefficient factor along ``axis`` and
    a ``ConstantSpace`` on every other factor (the ``Identity ⊗ D``
    extension), and reshapes ``leaf`` to broadcast along that axis.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the diagonal acts along.
    domain_factor : SpaceLike
        The coefficient factor the diagonal reads (its ``space``).
    codomain_factor : SpaceLike
        The coefficient factor the diagonal emits on.
    leaf : jax.Array
        The 1D per-mode diagonal along ``axis``.

    Returns
    -------
    Symbol
        The broadcast-tagged diagonal.
    """
    space = _lift(bare, axis, domain_factor)
    codomain = _lift(bare, axis, codomain_factor)
    ndim = len(bare.shape)
    index = bare.names.index(axis)
    data = axis_vector(jnp.asarray(leaf), ndim, index)
    return Symbol(space, data, codomain=codomain)


def _lift(
    bare: SpaceLike, axis: str, coeff_factor: SpaceLike,
) -> SpaceLike:
    """Product with ``coeff_factor`` at ``axis``, constants elsewhere."""
    if isinstance(bare, FunctionSpace):
        return coeff_factor
    factors = tuple(
        coeff_factor if axis in factor.names else factor.mesh.constant
        for factor in bare.factors)
    return TensorProductSpace.of(*factors)
