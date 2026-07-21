"""Parameterful hydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``, ``hor_divergence``) live on ``hy.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``hy.Core``
(the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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
    return state["p_hyd"].with_data(0.5 * (u**2 + v**2)).with_metadata(
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
    return state["b"].with_data(0.5 * b**2 / n2).with_metadata(
        name="epot", long_name="Potential energy", units="m^2/s^2")


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
}
