"""Parameterful nonhydrostatic diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics
(``rel_vort_z``) live on ``nh.State`` instead.

The ``DIAGNOSTICS`` mapping is contributed by ``nh.Core``
(the diagnostics-namespace channel); ``STRATIFICATION_DIAGNOSTICS``
by ``nh.ConstantStratification``, the module that carries the
background the total buoyancy adds back.
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
from fridom.nonhydro2.state import vorticity_corner
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
    return state["p"].new_quantity(
        0.5 * (u**2 + v**2 + dsqr * w**2),
        name="ekin", long_name="Kinetic energy", units="m^2/s^2")


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
    energy metric weight ``1/N^2_eff`` on ``b`` (``fr.model.EnergyMetric``);
    the effective ``N^2`` is assembled from the variant's primitives
    (:func:`_n2_eff`). The buoyancy is interpolated onto the pressure
    cell.
    """
    n2 = _n2_eff(params)
    center = state["p"].function_space
    b = state["b"].to(center).data
    return state["p"].new_quantity(
        0.5 * b**2 / n2,
        name="epot", long_name="Potential energy", units="m^2/s^2")


def etot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Total wave energy ``ekin + epot`` at cell center.

    Description
    -----------
    The sum of the linearized kinetic and available potential
    densities,

    .. math::
        E = \tfrac12 (u^2 + v^2 + \delta^2 w^2)
            + \tfrac12 \frac{b^2}{N^2_\mathrm{eff}},

    i.e. the quadratic energy of the metric ``M`` that the eigenmode
    and projection machinery norms with (``fr.model.EnergyMetric``), which
    the linear model (``advection=None``) conserves. Both parts are
    sampled at the cell centre, so the integral is a centre-sampled
    proxy of the ``M``-norm rather than that norm to machine
    precision. Integrate it and read the scalar with

    .. code-block:: python

        e = model.diagnostics.etot().integrate().item()
    """
    total = ekin(state, params) + epot(state, params)
    return state["p"].new_quantity(
        total.data,
        name="etot", long_name="Total energy", units="m^2/s^2")


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
    # b is BC-free, so its vertical derivative carries no declared
    # wall structure and the interpolation back to centers is the
    # explicit one-sided diagnostics closure (R2, boundary_plan.md)
    dbdz = _to_center_one_sided(
        state["b"].diff("z"), center).data
    # the vorticity pair does carry one: both differences retag onto
    # the shared free-slip edge (nh.State.rel_vort_z), which is what
    # lets them combine at all on a walled horizontal and grounds the
    # conversion to centers
    corner = vorticity_corner(state["u"], state["v"])
    zeta = (state["v"].diff("x").retag(corner)
            - state["u"].diff("y").retag(corner))
    q = f0 / n2 * dbdz + zeta.to(center).data
    if SCALING_NONLINEARITY in params:
        q = params[SCALING_NONLINEARITY] * q
    return state["p"].new_quantity(
        q, name="linear_pot_vort",
        long_name="Linear potential vorticity", units="1/s")


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
    vertical coordinate read off the field's own nodes, on whichever
    family ``b`` is declared. It is contributed by
    ``ConstantStratification`` rather than the core, since only that
    module carries a background.
    """
    b = state["b"]
    total = b + b.nodes("z") * _background_n2(params)
    return b.new_quantity(
        total.data,
        name="b_total", long_name="Total buoyancy", units="m/s^2")


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "etot": etot,
    "linear_pot_vort": linear_pot_vort,
}

#: contributed by ``nh.ConstantStratification`` (the background owner)
STRATIFICATION_DIAGNOSTICS = {
    "b_total": b_total,
}
