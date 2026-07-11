r"""
Factor-wise space-tag validators for the composition algebras.

Description
-----------
Owning class doc: ``notes/framework2/composition_refactor_plan.md``
(section 3, "Layer 0 — shared pure utilities" and section 5, stage
S0a). The **one** factor-wise tag validator shared by the diagonal /
realized-map algebras (``Symbol`` today; the realized maps next
stage). Two pure functions on interned coefficient spaces:

- :func:`compose_spaces` — the asymmetric ``@`` rule (the
  ``Identity ⊗ D`` tensor-product extension): the inner codomain met
  with the outer domain per factor, identity where both are non-
  ``Constant``, disjoint/passthrough where either is ``Constant``.
- :func:`union_spaces` — the symmetric ``+`` / ``*`` rule: identical
  or one ``Constant`` per factor.

Both preserve interning (``TensorProductSpace.of``) so their results
are the identical interned objects the inline reimplementations
produced. The module lives under ``grid/spaces`` — not
``grid/operators`` — so both the symbol algebra and ``base.py`` can
import it without a cycle (spaces must never import operators).
"""
# Wave: composition refactor S0a
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.tensor_product import TensorProductSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


# ================================================================
#  Symmetric union (``+`` / ``*``)
# ================================================================
def union_spaces(
    a: SpaceLike, b: SpaceLike, *, operation: str = "union",
) -> SpaceLike:
    r"""
    Factor-wise tag union (``Constant ⊗ X -> X``); mismatch raises.

    Description
    -----------
    The symmetric ``+`` / ``*`` / ``-`` / ``/`` rule: per factor the
    two tags must be identical (interned ``is``) or one must be
    ``ConstantSpace`` (the wildcard), in which case the other wins.
    Any disagreement on a non-``Constant`` factor — or a rank
    mismatch — raises :class:`SpaceMismatchError`. The result is
    rebuilt through ``TensorProductSpace.of`` so it is the identical
    interned object.

    Parameters
    ----------
    a : SpaceLike
        The left operand's coefficient space.
    b : SpaceLike
        The right operand's coefficient space.
    operation : str, optional
        The label recorded on a raised ``SpaceMismatchError``
        (default: "union").

    Returns
    -------
    SpaceLike
        The factor-wise union space.
    """
    if a is b:
        return a
    a_factors = a.factors
    b_factors = b.factors
    if len(a_factors) != len(b_factors):
        raise SpaceMismatchError(
            f"symbols on incompatible spaces {a!r} vs {b!r}",
            left=a, right=b, operation=operation)
    factors = []
    for fa, fb in zip(a_factors, b_factors, strict=True):
        if fa is fb:
            factors.append(fa)
        elif isinstance(fa, ConstantSpace):
            factors.append(fb)
        elif isinstance(fb, ConstantSpace):
            factors.append(fa)
        else:
            raise SpaceMismatchError(
                "symbols disagree on a non-constant factor: "
                f"{fa!r} vs {fb!r}", left=fa, right=fb,
                operation=operation)
    return TensorProductSpace.of(*factors)


# ================================================================
#  Asymmetric composition (``@``)
# ================================================================
def compose_spaces(
    inner_domain: SpaceLike,
    inner_codomain: SpaceLike,
    outer_domain: SpaceLike,
    outer_codomain: SpaceLike,
) -> tuple[SpaceLike, SpaceLike]:
    r"""
    Factor-wise composition tags for ``outer @ inner``.

    Description
    -----------
    The asymmetric ``@`` rule (the ``Identity ⊗ D`` tensor-product
    extension) for the composite ``outer ∘ inner``: per axis the
    shared physical mode index is the inner codomain met with the
    outer domain — required identical (interned ``is``) where **both**
    are non-``ConstantSpace`` (else :class:`SpaceMismatchError`),
    disjoint where either is ``Constant``. The composed domain factor
    is the inner domain (else the outer domain where the inner passes
    through), the composed codomain factor is the outer codomain (else
    the inner codomain where the outer passes through). Results are
    rebuilt through ``TensorProductSpace.of`` (identical interned
    objects).

    Parameters
    ----------
    inner_domain : SpaceLike
        Domain of the inner map (applied first).
    inner_codomain : SpaceLike
        Codomain of the inner map (the meeting point).
    outer_domain : SpaceLike
        Domain of the outer map (the meeting point).
    outer_codomain : SpaceLike
        Codomain of the outer map (applied second).

    Returns
    -------
    tuple[SpaceLike, SpaceLike]
        The ``(domain, codomain)`` of the composite.
    """
    inner_domain_f = inner_domain.factors
    inner_codomain_f = inner_codomain.factors
    outer_domain_f = outer_domain.factors
    outer_codomain_f = outer_codomain.factors
    if not (len(outer_domain_f) == len(inner_domain_f)
            == len(inner_codomain_f) == len(outer_codomain_f)):
        raise SpaceMismatchError(
            "cannot compose symbols on incompatible spaces "
            f"{outer_domain!r} @ {inner_domain!r}",
            left=outer_domain, right=inner_domain,
            operation="Symbol.__matmul__")
    domain_factors = []
    codomain_factors = []
    for fod, foc, fid, fic in zip(
            outer_domain_f, outer_codomain_f,
            inner_domain_f, inner_codomain_f, strict=True):
        shared_outer = not isinstance(fod, ConstantSpace)
        shared_inner = not isinstance(fic, ConstantSpace)
        if shared_outer and shared_inner and fod is not fic:
            raise SpaceMismatchError(
                "cannot compose symbols: inner codomain factor "
                f"{fic!r} != outer domain factor {fod!r}",
                left=fic, right=fod, operation="Symbol.__matmul__")
        domain_factors.append(
            fid if not isinstance(fid, ConstantSpace) else fod)
        codomain_factors.append(
            foc if not isinstance(foc, ConstantSpace) else fic)
    return (TensorProductSpace.of(*domain_factors),
            TensorProductSpace.of(*codomain_factors))
