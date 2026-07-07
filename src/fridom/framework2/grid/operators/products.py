"""
Pointwise product operators behind the field arithmetic dunders.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_products.md``.
Iteration-1 subset: the elementwise family the default table
registers on nodal and average spaces — ``CollocationProduct``
(``"multiply"``; on average spaces the documented second-order
shortcut), ``Divide`` (``"divide"``), ``Power`` (``"power"``), and
``Abs`` (``"abs"``, nodal-only by design). The operands reach these
operators already lifted onto their common space (doc 02's join +
sanctioned lifts), so the codomain resolvers demand identical bare
spaces. ``Where``/``Hadamard``/``Convolution``/``ConstantBroadcast``
land in later waves (no field sugar consumes them yet).
"""
# Wave 2: CollocationProduct, Divide, Power, Abs --
#    Wave 3+: Where, ConstantBroadcast, Hadamard, Convolution
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    BinaryOperator,
    FieldLike,
    UnaryOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


def _common_space(
    domain_a: SpaceLike, domain_b: SpaceLike, operation: str,
) -> SpaceLike:
    """Demand identical (pre-lifted) bare spaces; return them."""
    if domain_a is not domain_b:
        raise SpaceMismatchError(
            f"the operands of {operation!r} must share one space "
            f"after the sanctioned lifts, got {domain_a!r} vs "
            f"{domain_b!r}", left=domain_a, right=domain_b,
            operation=operation)
    return domain_a


def _elementwise(
    f: FieldLike, data: object, *others: FieldLike,
) -> FieldLike:
    """
    Build a default-metadata result on ``f``'s bare space.

    Description
    -----------
    Pointwise on aligned storage frames, so valid ghost slots stay
    valid: the result claims the pointwise minimum of the operands'
    halo validity (task 1.8, stage B).
    """
    valid = f.halo_valid
    for other in others:
        valid = valid.merge_min(other.halo_valid)
    return type(f)(f.grid, f.function_space.bare, data, None,
                   halo_valid=valid)


@final
@interned
class CollocationProduct(BinaryOperator):

    """
    Pointwise physical product on nodal spaces (aliased).

    Description
    -----------
    The ``*`` default on nodal spaces and — as the documented
    second-order shortcut identifying averages with midpoint values —
    on ``CellAvg``/``FaceAvg``. Aliased by design: dealiasing is the
    bracketing transforms' job (rules section 3.12). Every
    ``("multiply", *)`` default row holds one shared instance
    (registry form-2 resolution). halo 0, layout "any".
    """

    dispatch_kind: ClassVar[str | None] = "multiply"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(
        self, domain_a: SpaceLike, domain_b: SpaceLike,
    ) -> SpaceLike:
        """
        Return the common space of the (pre-lifted) operands.

        Parameters
        ----------
        domain_a : SpaceLike
            The left operand's bare space.
        domain_b : SpaceLike
            The right operand's bare space.

        Returns
        -------
        SpaceLike
            The shared bare space.
        """
        return _common_space(domain_a, domain_b, "multiply")

    def _apply(self, f: FieldLike, g: FieldLike) -> FieldLike:
        """
        Elementwise product of the aligned storage arrays.

        Parameters
        ----------
        f : FieldLike
            The first operand.
        g : FieldLike
            The second operand.

        Returns
        -------
        FieldLike
            The product field (default metadata: new quantity).
        """
        return _elementwise(
            f,
            f._data * g._data,  # noqa: SLF001 — storage seam
            g)


@final
@interned
class Divide(BinaryOperator):

    """
    Pointwise quotient on nodal/average spaces.

    Description
    -----------
    Same semantics as ``CollocationProduct`` (elementwise on nodal
    values, second-order shortcut on averages); no coefficient-space
    rows by design (a quotient of spectra has no
    representation-independent realization).
    """

    dispatch_kind: ClassVar[str | None] = "divide"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(
        self, domain_a: SpaceLike, domain_b: SpaceLike,
    ) -> SpaceLike:
        """
        Return the common space of the (pre-lifted) operands.

        Parameters
        ----------
        domain_a : SpaceLike
            The numerator's bare space.
        domain_b : SpaceLike
            The denominator's bare space.

        Returns
        -------
        SpaceLike
            The shared bare space.
        """
        return _common_space(domain_a, domain_b, "divide")

    def _apply(self, f: FieldLike, g: FieldLike) -> FieldLike:
        """
        Elementwise quotient of the aligned storage arrays.

        Parameters
        ----------
        f : FieldLike
            The numerator.
        g : FieldLike
            The denominator.

        Returns
        -------
        FieldLike
            The quotient field (default metadata: new quantity).
        """
        return _elementwise(
            f,
            f._data / g._data,  # noqa: SLF001 — storage seam
            g)


@final
@interned
class Power(BinaryOperator):

    """
    Pointwise power on nodal/average spaces.

    Description
    -----------
    Backs ``f ** p``: the physical power per the section-2.5 ``**``
    table. Scalar exponents reach it lifted onto the base's space
    (doc 02's dunder owns the lift), so the codomain is the common
    space unchanged.
    """

    dispatch_kind: ClassVar[str | None] = "power"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(
        self, domain_a: SpaceLike, domain_b: SpaceLike,
    ) -> SpaceLike:
        """
        Return the common space (scalar exponents keep it).

        Parameters
        ----------
        domain_a : SpaceLike
            The base field's bare space.
        domain_b : SpaceLike
            The (lifted) exponent's bare space.

        Returns
        -------
        SpaceLike
            The shared bare space.
        """
        return _common_space(domain_a, domain_b, "power")

    def _apply(self, f: FieldLike, p: FieldLike) -> FieldLike:
        """
        Elementwise ``f ** p`` on the aligned storage arrays.

        Parameters
        ----------
        f : FieldLike
            The base field.
        p : FieldLike
            The (lifted) exponent field.

        Returns
        -------
        FieldLike
            The power field (default metadata: new quantity).
        """
        return _elementwise(
            f,
            f._data ** p._data,  # noqa: SLF001 — storage seam
            p)


@final
@interned
class Abs(UnaryOperator):

    """
    Pointwise absolute value / complex modulus on nodal spaces.

    Description
    -----------
    The one scalar-changing pointwise op (``fr.Complex ->
    fr.Real``). Nodal-only by design: a sign change inside a cell
    makes ``|avg(u)| != avg(|u|)``, so an average-space shortcut
    would be silently wrong. Whole-space, not bindable.
    """

    dispatch_kind: ClassVar[str | None] = "abs"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        abs: S -> the fr.Real variant of S.

        Parameters
        ----------
        domain : SpaceLike
            The operand's bare space.

        Returns
        -------
        SpaceLike
            The per-factor ``fr.Real`` variant.
        """
        if isinstance(domain, TensorProductSpace):
            return TensorProductSpace.of(
                *(factor.as_real() for factor in domain.factors))
        return domain.as_real()

    def _apply(self, f: FieldLike) -> FieldLike:
        """
        Elementwise magnitude of the storage array.

        Parameters
        ----------
        f : FieldLike
            The operand field.

        Returns
        -------
        FieldLike
            The modulus field on the real space (default metadata).
        """
        codomain = self.codomain(f.function_space.bare)
        # pointwise on the storage frame: valid ghosts stay valid
        return type(f)(
            f.grid, codomain,
            jnp.abs(f._data),  # noqa: SLF001 — storage seam
            None, halo_valid=f.halo_valid)
