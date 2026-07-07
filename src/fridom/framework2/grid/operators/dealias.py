"""
Dealiasing helpers: ``PadFactor`` and the ``degree`` factory.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_transforms.md``
("PadFactor and ``dealias.degree``"; rules section 3.12). A
``PadFactor`` parameterizes the padded transform variants: with
``pad=degree(p)`` a transform's ``backward`` lands in the finer nodal
space of the ``refined(factor)`` mesh and its ``forward`` trims back
to the unpadded coefficient space (the 3/2 rule at ``p = 2``).
"""
# Wave 3: PadFactor, degree()
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class PadFactor:

    """
    Dealiasing pad factor for padded transform variants.

    Description
    -----------
    A static value consumed by ``Transform(pad=...)``: the exact
    refinement ratio of the padded nodal space (e.g. ``3/2`` for a
    quadratic nonlinearity). The transform calls
    ``mesh.refined(factor)`` once at construction.

    Parameters
    ----------
    factor : Fraction
        The refinement ratio of the padded nodal space; must be an
        exact ``Fraction`` (or int) >= 1.
    """

    #: refinement ratio of the padded nodal space, e.g. 3/2
    factor: Fraction

    def __post_init__(self) -> None:
        """Validate and normalize the ratio to an exact Fraction."""
        factor = self.factor
        if isinstance(factor, bool) or not isinstance(
                factor, int | Fraction):
            raise TypeError(
                "pad factors are exact Fractions (or ints), got "
                f"{factor!r}")
        factor = Fraction(factor)
        if factor < 1:
            raise ValueError(
                f"pad factors refine (factor >= 1), got {factor}")
        object.__setattr__(self, "factor", factor)


def degree(p: int) -> PadFactor:
    """
    Pad factor ``(p + 1) / 2`` for a degree-``p`` nonlinearity.

    Description
    -----------
    A degree-p product needs at least ``(p + 1)/2 * N`` points to be
    computed without aliasing (rules section 3.12); ``degree(2)`` is
    the classical 3/2 rule.

    Parameters
    ----------
    p : int
        The polynomial degree of the nonlinearity (>= 1).

    Returns
    -------
    PadFactor
        The pad factor ``(p + 1) / 2``.
    """
    if isinstance(p, bool) or not isinstance(p, int):
        raise TypeError(
            f"nonlinearity degrees are integers, got {p!r}")
    if p < 1:
        raise ValueError(
            f"nonlinearity degrees are positive, got {p}")
    return PadFactor(Fraction(p + 1, 2))
