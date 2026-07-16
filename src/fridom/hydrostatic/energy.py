r"""The hydrostatic energy-weight builder (the model's ``diag(M)``)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


def hydrostatic_energy_weights(
    inv_n2: float | ScalarField, inv_csqr: float,
) -> dict[str, float | ScalarField]:
    r"""Assemble the hydrostatic weights ``diag(1, 1, 1/N^2, 1/c^2)``.

    Description
    -----------
    The canonical hydrostatic energy metric ``M`` on the prognostic
    quadruple ``(u, v, b, ps)`` — the diagnosed ``w`` carries no
    independent energy (it is slaved to the horizontal divergence).
    The barotropic ``ps`` weight is ``1/c^2`` **integrated over the
    full depth**: on the ``Profile("x", "y")`` (constant-along-z)
    space ``ps`` broadcasts to the 3D volume before the quadrature, so
    its energy is the shallow-water barotropic potential energy
    ``(1/2) H ps^2 / c^2`` per unit area — the factor ``H`` that pairs
    the barotropic gravity term ``-grad ps`` (a 3D momentum forcing)
    with the depth-mean divergence ``-c^2 div(u_bar)`` into an exactly
    skew-adjoint operator.

    The caller passes the **already-computed** reciprocals
    ``inv_n2`` and ``inv_csqr`` so the degenerate ``N^2 = 0`` /
    ``c^2 = 0`` paths (which the eigenmode classes permit, falling
    back to ``1``) never divide here. A meridionally stratified model
    passes the reciprocal **profile field** ``1/N^2(y)`` — the ``b``
    weight is then sampled at the ``b`` nodes wherever the metric is
    applied, the pointwise pairing that keeps the buoyancy coupling
    M-skew for any profile.

    Parameters
    ----------
    inv_n2 : float | ScalarField
        The reciprocal squared buoyancy frequency ``1/N^2`` (the ``b``
        weight), computed by the caller; a profile field for a
        meridionally varying stratification.
    inv_csqr : float
        The reciprocal squared phase speed ``1/c^2`` (the ``ps``
        weight), computed by the caller.

    Returns
    -------
    dict[str, float | ScalarField]
        The ``(u, v, b, ps)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "b": inv_n2, "ps": inv_csqr}
