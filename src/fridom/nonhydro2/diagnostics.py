"""Parameterful nonhydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``) live on ``nh.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``nh.DynamicalCore``
(the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.params import (
    CORIOLIS_F0,
    SCALING_ROSSBY,
    STRATIFICATION_N2,
)
from fridom.nonhydro2.params import DSQR

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField


def ekin(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Kinetic energy ``0.5 (u^2 + v^2 + dsqr w^2)`` at cell center.

    Description
    -----------
    Carries ``dsqr`` (so it is a bound diagnostic, not a State
    property). Velocities are interpolated onto the pressure cell.
    """
    dsqr = params[DSQR]
    center = state["p"].function_space
    u = state["u"].to(center).data
    v = state["v"].to(center).data
    w = state["w"].to(center).data
    return state["p"].with_data(0.5 * (u**2 + v**2 + dsqr * w**2))


def epot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Potential energy ``0.5 b^2 / N^2`` at cell center.

    Description
    -----------
    The linearized (quadratic) potential energy consistent with the
    energy metric weight ``1/N^2`` on ``b`` (``fr.EnergyMetric``).
    Carries ``N^2``; the buoyancy is interpolated onto the pressure
    cell.
    """
    n2 = params[STRATIFICATION_N2]
    center = state["p"].function_space
    b = state["b"].to(center).data
    return state["p"].with_data(0.5 * b**2 / n2)


def linear_pot_vort(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Linear potential vorticity ``Ro (f0/N^2 d_z b + zeta_z)``.

    Description
    -----------
    The linearized Ertel PV (the old ``linear_pot_vort``). Carries
    ``f0``, ``N^2`` and the Rossby number.
    """
    f0 = params[CORIOLIS_F0]
    n2 = params[STRATIFICATION_N2]
    ro = params[SCALING_ROSSBY]
    center = state["p"].function_space
    dbdz = state["b"].diff("z").to(center).data
    zeta = (state["v"].diff("x") - state["u"].diff("y")).to(center)
    return state["p"].with_data(ro * (f0 / n2 * dbdz + zeta.data))


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "linear_pot_vort": linear_pot_vort,
}
