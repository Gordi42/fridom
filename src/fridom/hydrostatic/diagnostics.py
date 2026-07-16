"""Parameterful hydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``, ``hor_divergence``) live on ``hy.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``hy.HydrostaticCore``
(the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.params import STRATIFICATION_N2

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


def epot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Potential energy ``0.5 b^2 / N^2`` at cell center.

    Description
    -----------
    The linearized (quadratic) potential energy consistent with the
    energy metric weight ``1/N^2`` on ``b`` (``hy.energy``). Carries
    ``N^2``; the buoyancy is already collocated with the pressure
    cell.
    """
    n2 = params[STRATIFICATION_N2]
    b = state["b"].data
    return state["b"].with_data(0.5 * b**2 / n2).with_metadata(
        name="epot", long_name="Potential energy", units="m^2/s^2")


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
}
