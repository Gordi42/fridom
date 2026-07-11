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
from fridom.spatial.operators.interp import LinearInterp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _to_center_one_sided(
    field: ScalarField, center: SpaceLike,
) -> ScalarField:
    """
    Interpolate onto the center space, one-sided at BC-free walls.

    Description
    -----------
    Diagnostics interpolate staggered derivative outputs whose
    wall-normal factors carry no declared BC structure (the PV's
    ``d_z b``, the vorticity's mixed-tag meet). Diagnostics are
    exactly the sanctioned use of the explicit one-sided closure
    (R2, boundary_plan.md): interior windows unchanged, wall-window
    outputs recomputed from true DOFs only. On periodic or
    BC-tagged axes the variant is identical to the default row.

    Parameters
    ----------
    field : ScalarField
        The (possibly staggered) diagnostic field.
    center : SpaceLike
        The target cell-center product space.

    Returns
    -------
    ScalarField
        The field interpolated onto ``center``.
    """
    interp = LinearInterp(boundary="one_sided")
    out = field
    for axis in center.bare.names:
        if (out.function_space.bare.factor(axis)
                is not center.bare.factor(axis)):
            out = interp[axis](out)
    return out


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
    # the derivative outputs carry no declared wall structure (b is
    # BC-free; the vorticity difference joins to the BC-free meet),
    # so the interpolation back to centers is the explicit one-sided
    # diagnostics closure (R2, boundary_plan.md)
    dbdz = _to_center_one_sided(
        state["b"].diff("z"), center).data
    zeta = _to_center_one_sided(
        state["v"].diff("x") - state["u"].diff("y"), center)
    return state["p"].with_data(ro * (f0 / n2 * dbdz + zeta.data))


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "linear_pot_vort": linear_pot_vort,
}
