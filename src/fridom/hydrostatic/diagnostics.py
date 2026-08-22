"""Parameterful hydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``, ``hor_divergence``) live on ``hy.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``hy.Core``
(the diagnostics-namespace channel); ``STRATIFICATION_DIAGNOSTICS``
by ``hy.ConstantStratification``, the module that carries the
background the total buoyancy adds back.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.hydrostatic.params import GRAVITY
from fridom.model.params import (
    SCALING_NONLINEARITY,
    STRATIFICATION_FROUDE,
    STRATIFICATION_N2,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField


def ekin(
    state: VectorField, params: Mapping[str, object],  # noqa: ARG001
) -> ScalarField:
    """Horizontal kinetic energy ``0.5 (u^2 + v^2)`` at cell center.

    Description
    -----------
    The hydrostatic kinetic energy is horizontal only (there is no
    prognostic vertical momentum). The face velocities are
    interpolated onto the pressure cell.
    """
    center = state["p_hyd"].function_space
    u = state["u"].to(center).data
    v = state["v"].to(center).data
    return state["p_hyd"].new_quantity(
        0.5 * (u**2 + v**2),
        name="ekin", long_name="Kinetic energy", units="m^2/s^2")


def _n2_eff(params: Mapping[str, object]) -> object:
    r"""Return the effective ``N^2`` from the variant's primitives.

    Dimensional: the ``stratification.n2`` provide. Nondimensional:
    :math:`(\varepsilon/\mathrm{Fr})^2` (the live internal-wave
    ratio; self-normalizing under the matching ``InternalWave``
    scaling).
    """
    if STRATIFICATION_FROUDE in params:
        ratio = (params[SCALING_NONLINEARITY]
                 / params[STRATIFICATION_FROUDE])
        return ratio * ratio
    return params[STRATIFICATION_N2]


def _background_n2(params: Mapping[str, object]) -> object:
    r"""Return the vertical gradient of the background buoyancy.

    Dimensional: the ``stratification.n2`` provide, the background
    being :math:`N^2 z`. Nondimensional: :math:`\varepsilon/\mathrm{Fr}^2`,
    the effective :math:`N^2_\mathrm{eff} = (\varepsilon/\mathrm{Fr})^2`
    of the restoring term divided by the nonlinearity
    :math:`\varepsilon` that the advection of the nondimensional
    equations carries, so that anomaly plus background is advected as
    one field (the ``b`` unit row then converts it to :math:`N^2 z`).
    """
    if STRATIFICATION_FROUDE in params:
        froude = params[STRATIFICATION_FROUDE]
        return params[SCALING_NONLINEARITY] / (froude * froude)
    return params[STRATIFICATION_N2]


def epot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Potential energy ``0.5 b^2 / N^2_eff`` at cell center.

    Description
    -----------
    The linearized (quadratic) potential energy consistent with the
    energy metric weight ``1/N^2_eff`` on ``b`` (``hy.energy``); the
    effective ``N^2`` is assembled from the variant's primitives
    (:func:`_n2_eff`). The buoyancy is already collocated with the
    pressure cell.
    """
    n2 = _n2_eff(params)
    b = state["b"].data
    return state["b"].new_quantity(
        0.5 * b**2 / n2,
        name="epot", long_name="Potential energy", units="m^2/s^2")


def eta(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Free-surface elevation ``eta = p_s / g`` (dimensional).

    Description
    -----------
    The surface displacement in metres, from the surface pressure the
    model carries as ``p_s = g\,eta``. It reads the dimensional
    ``hydrostatic.gravity`` provide, so it is defined on the
    **dimensional** variant only (the nondimensional core carries no
    gravity — read ``ps`` directly there). ``ps`` is broadcast from
    the ``Profile("x", "y")`` space, so the returned field is
    constant along ``z``.
    """
    gravity = params[GRAVITY]
    ps = state["ps"]
    return ps.new_quantity(
        ps.data / gravity,
        name="eta", long_name="Surface elevation", units="m")


def b_total(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Total buoyancy ``b + N^2 z`` at the buoyancy nodes.

    Description
    -----------
    The prognostic ``b`` of a ``ConstantStratification`` model is the
    departure from the background :math:`N^2 z` that the module
    carries, so the isopycnals of the water column belong to the sum
    of the two. This adds the background back, its gradient assembled
    from the variant's primitives (:func:`_background_n2`) and the
    vertical coordinate read off the field's own nodes. It is
    contributed by ``ConstantStratification`` rather than the core,
    since only that module carries a background.
    """
    b = state["b"]
    total = b + b.nodes("z") * _background_n2(params)
    return b.new_quantity(
        total.data,
        name="b_total", long_name="Total buoyancy", units="m/s^2")


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "eta": eta,
}

#: contributed by ``hy.ConstantStratification`` (the background owner)
STRATIFICATION_DIAGNOSTICS = {
    "b_total": b_total,
}
