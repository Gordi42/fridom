r"""Parameterful shallow-water diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics (``rel_vort``,
``divergence``) live on ``sw.State`` instead.

Two energy families
-------------------
The package exposes **both** energies, and they are different
functionals — the user picks:

- ``ekin`` / ``epot`` — the **linearized** (quadratic) energy
  densities of the energy metric ``M = diag(1, 1, 1/c^2)`` on
  ``(u, v, p)`` (``fr.EnergyMetric``; a single source of truth for
  the metric weights), sampled at the cell centre. ``M`` is the norm
  of the eigenmode / projection machinery, and the **linear** model
  (``advection=False``) conserves it exactly — but exactly means the
  ``M``-norm itself, the quadratics summed on the fields' *own*
  staggered spaces; these densities square the centre-*interpolated*
  velocities, so their integral is a centre-sampled proxy of that
  norm (they agree for resolved fields, not to machine precision).
  Either way they are **not** the invariant of the **nonlinear**
  model: with the Sadourny advection switched on, the scheme
  produces them at :math:`O(\mathrm{Ro})`.
- ``ekin_full`` / ``epot_full`` / ``etot_full`` — the
  **thickness-weighted (nonlinear)** energy that the Sadourny scheme
  plus the core's gravity term conserve **exactly** (semi-discretely,
  to machine precision, on periodic, walled and chart grids; see
  ``sw.modules.SadournyAdvection``). ``etot_full`` is the model's
  invariant — the quantity to print in a nonlinear run.

Every diagnostic returns a density at the cell centre; integrate it
with ``field.integrate()`` (which carries the metric Jacobian on any
mapped grid — the ``sqrt(g)`` area element on an embedding ``chart=``
grid, the column Jacobian on an analytic ``maps=`` grid) and read the
scalar with ``.item()``:

.. code-block:: python

    e = model.diagnostics.etot_full().integrate().item()

The ``DIAGNOSTICS`` mapping is contributed by ``sw.Core``
(the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.shallowwater2.params import CSQR, ROSSBY

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField


def ekin(
    state: VectorField,
    params: Mapping[str, object],  # noqa: ARG001 — diagnostic protocol
) -> ScalarField:
    """Kinetic energy ``0.5 (u^2 + v^2)`` at cell center.

    Description
    -----------
    The linearized (quadratic) kinetic energy matching the energy
    metric weights ``1`` on ``u`` and ``v``. Velocities are
    interpolated onto the pressure cell. The state components are the
    **physical** (m/s) velocities on every grid
    (``physical_state_components.md`` ruling (c)), so the chart branch
    collapses to this flat spelling — the physical speed squared needs
    no metric root (D4).
    """
    center = state["p"].function_space
    u = state["u"].to(center)
    v = state["v"].to(center)
    return state["p"].with_data(0.5 * (u.data**2 + v.data**2))


def epot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Potential energy ``0.5 p^2 / c^2`` at cell center.

    Description
    -----------
    The linearized (quadratic) potential energy consistent with the
    energy metric weight ``1/c^2`` on ``p``. Carries ``c^2``.
    """
    csqr = params[CSQR]
    p = state["p"]
    return p.with_data(0.5 * p.data**2 / csqr)


# ================================================================
#  The thickness-weighted (nonlinear) energy — the model's invariant
# ================================================================
def thickness(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Full geopotential thickness ``h = c^2 + Ro p`` at cell center.

    Description
    -----------
    The same ``p_full`` the Sadourny advection and the core's flux
    form carry (the ``csqr`` **field**, never the scalar — so a
    variable-depth model is spatially correct).
    """
    rossby = params[ROSSBY]
    p = state["p"]
    return state["csqr"].to(p) + rossby * p


def ekin_full(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Thickness-weighted kinetic energy density at cell center.

    Description
    -----------
    The kinetic part of the energy the scheme **actually conserves**:

    .. math::
        E_\mathrm{kin} = \int \tfrac12 \bar{h}^x u^2
            + \tfrac12 \bar{h}^y v^2 , \qquad
        h = c^2 + \mathrm{Ro}\, p

    with each thickness average :math:`\bar h` taken **at the
    velocity's own node** (``h.to(u)`` / ``h.to(v)``) — that
    placement is what makes the semi-discrete conservation exact, so
    it is lifted verbatim from the scheme (see the module docstring
    of ``sw.modules.SadournyAdvection``), not re-derived.

    On chart grids the state components are the **physical** (m/s)
    velocities (``physical_state_components.md`` ruling (c)), so the
    quadratics carry only the :math:`\sqrt{g}` Jacobian **on each
    velocity's own staggered space** (no :math:`g_{ii}` factor, D4),
    interpolated to the centre and divided by the centre
    :math:`\sqrt{g}` — exactly the placement of the scheme's own
    Bernoulli kinetic energy:

    .. math::
        E_\mathrm{kin} = \int \frac{
            \overline{\sqrt{g}\,\bar h\, U^2}
            + \overline{\sqrt{g}\,\bar h\, V^2}}{2\sqrt{g}}

    so that ``.integrate()`` (which re-applies the centre
    :math:`\sqrt{g}`) reproduces the per-velocity sums of the
    invariant. Unlike ``ekin``, this is **not** the linearized
    quadratic; ``ekin`` is not conserved by the nonlinear model.
    """
    u, v, p = state["u"], state["v"], state["p"]
    grid = u.grid
    h = thickness(state, params)
    e_u = u * u * h.to(u)
    e_v = v * v * h.to(v)
    if grid.chart_coords is None:
        return p.with_data(0.5 * (e_u.to(p) + e_v.to(p)).data)
    e_u = grid.metric(u.function_space.bare, "sqrt_g") * e_u
    e_v = grid.metric(v.function_space.bare, "sqrt_g") * e_v
    sqrt_g = grid.metric(p.function_space.bare, "sqrt_g")
    return p.with_data(
        (0.5 * (e_u.to(p) + e_v.to(p)) / sqrt_g).data)


def epot_full(
    state: VectorField,
    params: Mapping[str, object],  # noqa: ARG001 — diagnostic protocol
) -> ScalarField:
    r"""Available potential energy ``0.5 p^2`` at cell center.

    Description
    -----------
    The potential part of the conserved (thickness-weighted) energy:
    :math:`\int \tfrac12 p^2` — **no** ``1/c^2`` (that weight belongs
    to the linearized ``epot``). It is the available part of the full
    potential energy :math:`\tfrac12 h^2/\mathrm{Ro}^2 =
    \mathrm{const} + c^2 p/\mathrm{Ro} + \tfrac12 p^2`, whose other
    two terms are fixed by (exact) mass conservation.
    """
    p = state["p"]
    return p.with_data(0.5 * p.data**2)


def etot_full(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Return the model's invariant: ``ekin_full + epot_full``.

    Description
    -----------
    The total thickness-weighted energy density,

    .. math::
        E = \int \tfrac12 \bar{h}^x u^2 + \tfrac12 \bar{h}^y v^2
            + \tfrac12 p^2

    (metric form on chart grids: ``ekin_full``). The core's gravity
    term and the Sadourny advection conserve it **exactly** in the
    semi-discrete sense — the split mass flux :math:`u h` makes the
    invariant belong to that pair, on periodic, walled and chart
    grids alike — so ``etot_full().integrate()`` is the quantity to
    monitor in a nonlinear run; the residual drift of a run is the
    time stepper's, not the scheme's.

    The remaining terms are exactly skew under the **linearized**
    metric ``M`` instead, and only bounded here (both recorded in
    ``sw.modules.SadournyAdvection``): the split Coriolis module
    (``f`` outside the potential vorticity) commits an
    :math:`O(\mathrm{Ro})` commutator error in this functional, and a
    prescribed ``background=`` flow exchanges energy with it. So the
    machine-precision statement is: **gravity + Sadourny conserve**
    the thickness-weighted energy; **Coriolis is skew under ``M``**
    (the linearized family).

    The Coriolis error is optional: assembling
    ``sw.modules.CoriolisEnergyCorrection`` next to the linear
    Coriolis module (or using the conserving
    ``sw.modules.NonlinearFPlaneCoriolis`` family instead of it)
    carries the rotation as the ``f``-part of the vector-invariant PV
    flux, and this functional is then conserved to machine precision
    **including** rotation — on flat, walled and chart grids
    (``sw.modules.coriolis``).
    """
    kin = ekin_full(state, params)
    return kin + epot_full(state, params)


# ================================================================
#  Potential vorticity — the scheme's materially-conserved tracer
# ================================================================
def pot_vort(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    r"""Potential vorticity ``q = (f + Ro zeta) / h`` at the corner.

    Description
    -----------
    The shallow-water potential vorticity, on the north-east
    vorticity corner (``sw.State.rel_vort``'s space):

    .. math::
        q = \frac{f + \mathrm{Ro}\,\zeta}{h} , \qquad
        h = c^2 + \mathrm{Ro}\,p ,

    with :math:`\zeta` the metric-aware relative vorticity
    (``sw.State.rel_vort`` — the :math:`\sqrt g`-weighted circulation
    of the *physical* components on a chart grid, not a flat
    cross-derivative), :math:`f` the Coriolis parameter read from the
    carried ``f_coriolis`` field (so it tracks an ``f(y)`` beta plane,
    the lat-lon sphere, and a time-dependent :math:`f`), and :math:`h`
    the full geopotential thickness (``thickness``). Carries the
    Rossby number and (through ``thickness``) :math:`c^2`.

    This is the **vector-invariant** potential vorticity that the
    Sadourny advection plus the conserving Coriolis flux actually
    transport (the combined :math:`(f + \mathrm{Ro}\,\zeta)/h`,
    ``sw.modules.coriolis``): it is the scheme's materially-conserved
    tracer, so a nonlinear run advects its extrema to advective-scheme
    tolerance (unlike the quadratic energy, which the split Coriolis
    produces at :math:`O(\mathrm{Ro})`).

    .. note::

        The Rossby factor on :math:`\zeta` is the scaling delta from
        the old-stack ``sw.State.pot_vort`` (which spelled the
        unscaled :math:`(\zeta + f)/h`): the two agree at
        :math:`\mathrm{Ro} = 1`, but only the scaled form here is the
        model's material invariant (the old form drifts at
        :math:`O(\mathrm{Ro} - 1)` in a nonlinear run).

    Parameters
    ----------
    state : VectorField
        The shallow-water state; reads ``u``, ``v``, ``p``, ``csqr``
        and ``f_coriolis`` (through ``rel_vort`` and ``thickness``).
    params : Mapping[str, object]
        The bound parameters; reads ``scaling.nonlinearity`` and, through
        ``thickness``, ``shallowwater.csqr``.

    Returns
    -------
    ScalarField
        The potential vorticity on the vorticity corner.
    """
    rossby = params[ROSSBY]
    zeta = state.rel_vort
    corner = zeta.function_space
    f = state["f_coriolis"].to(corner)
    h = thickness(state, params).to(corner)
    q = (f + rossby * zeta) / h
    return q.with_metadata(
        name="pot_vort", long_name="Potential vorticity",
        units="s/m^2")


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
    "ekin_full": ekin_full,
    "epot_full": epot_full,
    "etot_full": etot_full,
    "thickness": thickness,
    "pot_vort": pot_vort,
}
