"""Parameterful nonhydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``) live on ``nh.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``nh.Core``
(the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.params import (
    CORIOLIS_F0,
    CORIOLIS_ROSSBY,
    SCALING_NONLINEARITY,
    STRATIFICATION_FROUDE,
    STRATIFICATION_N2,
)
from fridom.nonhydro2.params import ASPECT_RATIO
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
    """Kinetic energy ``0.5 (u^2 + v^2 + delta^2 w^2)`` at cell center.

    Description
    -----------
    Carries the aspect ratio (so it is a bound diagnostic, not a
    State property; ``nonhydro.aspect_ratio``, squared at this use
    site). Velocities are interpolated onto the pressure cell.
    """
    delta = params[ASPECT_RATIO]
    dsqr = delta * delta
    center = state["p"].function_space
    u = state["u"].to(center).data
    v = state["v"].to(center).data
    w = state["w"].to(center).data
    return state["p"].with_data(0.5 * (u**2 + v**2 + dsqr * w**2))


def _n2_eff(params: Mapping[str, object]) -> object:
    r"""Return the effective ``N^2`` from the variant's primitives.

    Dimensional: the ``stratification.n2`` provide. Nondimensional:
    :math:`(\varepsilon/\mathrm{Fr})^2` (the live internal-wave
    ratio; under the matching ``InternalWave`` scaling it
    self-normalizes to 1.0).
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
    energy metric weight ``1/N^2_eff`` on ``b`` (``fr.EnergyMetric``);
    the effective ``N^2`` is assembled from the variant's primitives
    (:func:`_n2_eff`). The buoyancy is interpolated onto the pressure
    cell.
    """
    n2 = _n2_eff(params)
    center = state["p"].function_space
    b = state["b"].to(center).data
    return state["p"].with_data(0.5 * b**2 / n2)


def linear_pot_vort(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Linear potential vorticity ``eps (f_eff/N^2_eff d_z b + zeta_z)``.

    Description
    -----------
    The linearized Ertel PV, variant-aware from the primitives:
    dimensional ``f0/N^2 d_z b + zeta_z`` (no scaling factor at all);
    nondimensional ``eps (f_eff/N^2_eff d_z b + zeta_z)`` with
    ``f_eff = eps/Ro`` (a nondim f-plane) and
    ``N^2_eff = (eps/Fr)^2``.
    """
    if CORIOLIS_ROSSBY in params:
        f0 = (params[SCALING_NONLINEARITY]
              / params[CORIOLIS_ROSSBY])
    else:
        f0 = params[CORIOLIS_F0]
    n2 = _n2_eff(params)
    center = state["p"].function_space
    # the derivative outputs carry no declared wall structure (b is
    # BC-free; the vorticity difference joins to the BC-free meet),
    # so the interpolation back to centers is the explicit one-sided
    # diagnostics closure (R2, boundary_plan.md)
    dbdz = _to_center_one_sided(
        state["b"].diff("z"), center).data
    zeta = _to_center_one_sided(
        state["v"].diff("x") - state["u"].diff("y"), center)
    q = f0 / n2 * dbdz + zeta.data
    if SCALING_NONLINEARITY in params:
        q = params[SCALING_NONLINEARITY] * q
    return state["p"].with_data(q)


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "linear_pot_vort": linear_pot_vort,
}
