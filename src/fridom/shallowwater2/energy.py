r"""The shallow-water energy-weight builders (the model's ``diag(M)``)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


def shallowwater_energy_weights(inv_csqr: float) -> dict[str, float]:
    r"""Assemble the shallow-water weights ``diag(1, 1, 1/c^2)``.

    Description
    -----------
    The canonical shallow-water energy metric ``M`` on ``(u, v, p)``.
    The caller passes the **already-computed** reciprocal ``inv_csqr``
    so the degenerate ``c^2 = 0`` path (which the eigenmode class
    permits, falling back to ``1``) never divides here.

    Parameters
    ----------
    inv_csqr : float
        The reciprocal squared phase speed ``1/c^2`` (the ``p``
        weight), computed by the caller.

    Returns
    -------
    dict[str, float]
        The ``(u, v, p)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "p": inv_csqr}


def shallowwater_varying_energy_weights(
    csqr: ScalarField,
) -> dict[str, float | ScalarField]:
    r"""Assemble the variable-depth weights ``diag(c^2, c^2, 1)``.

    Description
    -----------
    The shallow-water energy metric for a **varying** :math:`c^2(y)`
    profile field: :math:`c^2` weights the *velocities* (sampled
    ``csqr.to(u)`` / ``csqr.to(v)`` wherever the metric is applied),
    not the pressure — because :math:`c^2` sits inside the
    divergence flux ``dp = -div(c^2 u)``, skewness pairs
    :math:`\langle c^2 u, -\nabla p\rangle` with
    :math:`\langle p, -\mathrm{div}(c^2 u)\rangle` through the
    discrete div/grad transposes, with the weight sampled exactly
    where the flux samples it. For a constant :math:`c^2` this
    convention differs from ``diag(1, 1, 1/c^2)`` by the overall
    factor :math:`c^2` only (frequencies and projectors agree).

    Parameters
    ----------
    csqr : ScalarField
        The squared phase-speed profile field (strictly positive).

    Returns
    -------
    dict[str, float | ScalarField]
        The ``(u, v, p)`` energy weights.
    """
    return {"u": csqr, "v": csqr, "p": 1.0}
