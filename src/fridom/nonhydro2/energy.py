r"""The nonhydrostatic energy-weight builder (the model's ``diag(M)``)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


def nonhydro_energy_weights(
    dsqr: float, inv_n2: float | ScalarField,
) -> dict[str, float | ScalarField]:
    r"""Assemble the nonhydro energy weights ``diag(1, 1, dsqr, 1/N^2)``.

    Description
    -----------
    The canonical nonhydro energy metric ``M`` on ``(u, v, w, b)``. The
    caller passes the **already-computed** reciprocal ``inv_n2`` so the
    degenerate ``N^2 = 0`` path (which the eigenmode classes permit,
    falling back to ``1``) never divides here. A meridionally
    stratified model passes the reciprocal **profile field**
    ``1/N^2(y)`` — the ``b`` weight is then sampled at the ``b``
    nodes wherever the metric is applied, the pointwise pairing that
    keeps the buoyancy coupling M-skew for any profile.

    Parameters
    ----------
    dsqr : float
        The squared aspect ratio (the ``w`` weight).
    inv_n2 : float | ScalarField
        The reciprocal squared buoyancy frequency ``1/N^2`` (the ``b``
        weight), computed by the caller; a profile field for a
        meridionally varying stratification.

    Returns
    -------
    dict[str, float | ScalarField]
        The ``(u, v, w, b)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "w": dsqr, "b": inv_n2}
