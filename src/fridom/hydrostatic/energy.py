r"""The hydrostatic energy-weight builder (the model's ``diag(M)``)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


def hydrostatic_energy_weights(
    inv_n2: float | ScalarField, ps_weight: float,
) -> dict[str, float | ScalarField]:
    r"""Assemble the hydrostatic weights ``diag(1, 1, 1/N^2, H/c^2)``.

    Description
    -----------
    The canonical hydrostatic energy metric ``M`` on the prognostic
    quadruple ``(u, v, b, ps)`` — the diagnosed ``w`` carries no
    independent energy (it is slaved to the horizontal divergence).
    ``fr.model.EnergyMetric.from_model(model)`` derives these weights
    from an assembled model and is the default spelling; this builder
    is for assembling them by hand (an unstratified model, a profile
    ``1/N^2``).

    The barotropic ``ps`` weight is the **depth-integrated** ``H/c^2``,
    which is ``1/g`` on the dimensional model (``c^2 = g H``). ``ps``
    lives on the ``Profile("x", "y")`` space and the metric's
    quadrature (``ScalarField.integrate``) reduces its ``ConstantSpace``
    z factor as the identity — an area integral, no column depth — so
    the depth rides in the weight, and the ``ps`` energy is the
    shallow-water barotropic potential energy
    ``(1/2) H ps^2 / c^2 = (1/2) ps^2 / g`` per unit area. That is the
    factor pairing the barotropic gravity term ``-grad ps`` (a 3D
    momentum forcing) with the depth-mean divergence
    ``-c^2 div(u_bar)`` into an exactly skew-adjoint operator; ``1/c^2``
    alone is off by the depth on any non-unit column.

    The caller passes the **already-computed** reciprocals ``inv_n2``
    and ``ps_weight`` so a degenerate ``N^2 = 0`` never divides here. A
    meridionally stratified model passes the reciprocal **profile
    field** ``1/N^2(y)`` — the ``b`` weight is then sampled at the ``b``
    nodes wherever the metric is applied, the pointwise pairing that
    keeps the buoyancy coupling M-skew for any profile.

    Parameters
    ----------
    inv_n2 : float | ScalarField
        The reciprocal squared buoyancy frequency ``1/N^2`` (the ``b``
        weight), computed by the caller; a profile field for a
        meridionally varying stratification.
    ps_weight : float
        The depth-integrated reciprocal squared phase speed ``H/c^2``
        (the ``ps`` weight), ``1/g`` on the dimensional model, computed
        by the caller.

    Returns
    -------
    dict[str, float | ScalarField]
        The ``(u, v, b, ps)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "b": inv_n2, "ps": ps_weight}
