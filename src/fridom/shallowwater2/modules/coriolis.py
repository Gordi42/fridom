r"""
The exactly-conserving (thickness-ratio) shallow-water Coriolis.

Description
-----------
The Sadourny advection plus the core's gravity term conserve the
**thickness-weighted** energy

.. math::
    E = \sum \tfrac12 \bar h^x u^2 + \tfrac12 \bar h^y v^2
        + \tfrac12 p^2 , \qquad h = c^2 + \mathrm{Ro}\,p

to machine precision (``sw.diagnostics.etot_full``). The *linear*
Coriolis module does not: it is exactly skew under the **linearized**
metric :math:`M`, not under the thickness-weighted norm, and the
mismatch produces :math:`E` at :math:`O(\mathrm{Ro})` (measured: 3.7e-2
relative on the sphere).

The exactly-conserving discrete Coriolis term is the ``f``-part of the
vector-invariant potential-vorticity flux — the *same* corner averages
the Sadourny scheme uses, with :math:`\zeta` replaced by :math:`f`:

.. math::
    \partial_t u = +\,\overline{F^v\,q_f}^{\,y} , \qquad
    \partial_t v = -\,\overline{F^u\,q_f}^{\,x} , \qquad
    q_f = \frac{\overline{f}^{\,xy}}{\overline{h}^{\,xy}} ,

with the corner mass fluxes :math:`F^u = \overline{u\,\bar h^x}`,
:math:`F^v = \overline{v\,\bar h^y}` (the NE vorticity corner, both
interpolated exactly as ``SadournyAdvection`` interpolates them). It
is a **ratio of averages** — nonlinear in ``h`` — and that is the whole
difficulty: continuously :math:`(f/h)\,(h v) = f v` collapses, but the
discrete averages do not cancel.

**Why it conserves, exactly.** The measure-weighted ``.to``
interpolations are adjoints of each other, so for *any* corner scalar
:math:`Q` the pair :math:`\partial_t u = \overline{F^v Q}`,
:math:`\partial_t v = -\overline{F^u Q}` produces

.. math::
    \sum \bar h^x u\,\partial_t u + \sum \bar h^y v\,\partial_t v
        = \langle F^u, F^v Q\rangle - \langle F^v, F^u Q\rangle = 0

— exactly the antisymmetric vorticity-flux exchange the Sadourny
docstring records (which holds "for any finite corner ``q``"). The
term advances only ``u`` and ``v``, so it does not move ``h`` either:
its production of the thickness-weighted energy is machine zero on
flat, walled and chart grids. Choosing :math:`Q = \bar f/\bar h` is
what makes it *consistent* with :math:`f v`; conservation itself is
free.

On a **chart** grid every average carries the metric exactly as
``SadournyAdvection`` places it: the corner mass fluxes are the
:math:`\sqrt{g}`-weighted fluxes :math:`F^i = \overline{\sqrt g\,\bar
h\,u^i}` that the thickness divergence carries, the momentum
tendencies are **covariant** and are raised (``raise_index``) onto the
prognostic contravariant components — the placement that makes the
exchange antisymmetric under
:math:`E = \sum \sqrt g\,\bar h\,g_{ii}(u^i)^2/2 + \dots`.

Two routes to the same tendency
-------------------------------
The expression above is one thing; there are two honest ways to put it
into a model, and which one is right depends on whether the run needs
the **linear operator** ``L`` (the ``linear=True`` terms):

- :class:`CoriolisEnergyCorrection` (**route A**, the default path) —
  an *optional* ``linear=False`` module carrying the **difference**
  ``(f/h_bar)(h v)_bar - f v``, assembled **alongside** the ordinary
  linear Coriolis module. The sum of the two is the conserving form,
  while ``L`` is bit-for-bit unchanged (the correction is declared
  nonlinear, and it vanishes identically at the rest state), so
  eigenmodes, projections, optimal balance and IMEX-by-linearity keep
  working exactly as today.
- :class:`NonlinearFPlaneCoriolis` / :class:`NonlinearBetaPlaneCoriolis`
  / :class:`NonlinearRotationCoriolis` (**route B**) — the conserving
  term carried **whole** in one ``linear=False`` module, used
  *instead* of the linear Coriolis module. Simpler and cheaper (one
  term, no cancellation), but ``L`` then has **no rotation at all**,
  which makes every ``L``-consumer invalid — so these modules declare
  a ``linear_operator_gap`` and the machinery refuses them
  (``fr.model.require_linear_operator``).

The two routes produce the **same tendency to rounding** (gated in
``tests/shallowwater2/test_coriolis.py``); they differ only in how it
is *declared*, and therefore in what the term-filtering machinery can
still do with the model afterwards. Assembling both — or route B next
to any linear Coriolis — double-counts rotation and is a taught error
at bind.

**The one thing route A does not restore** (recorded, small, benign):
where ``f`` varies (beta plane, sphere), the conserving form averages
``f`` to the *corner* while the linear module samples it at the ``u``
faces, so the correction is not *identically* zero as an operator on
velocity perturbations — its linearization about the rest state is a
residual of size :math:`O(\Delta^2 f'')`. That residual is itself
exactly M-skew (a difference of two M-skew rotations), so it does no
work and does not corrupt any ``L``-consumer; ``L`` remains a
consistent linearization of the scheme, to the scheme's own order. On
the f-plane (constant ``f``) the residual is zero to rounding.
"""
from __future__ import annotations

import jax.numpy as jnp

from fridom.model.declarations import FieldReference
from fridom.model.module import Module
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
    RotationCoriolis,
    chart_rotation,
    linear_rotation,
)
from fridom.model.parameters import Param
from fridom.model.params import SCALING_ROSSBY
from fridom.model.terms import term
from fridom.model.time_dependent import TimeDependent
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance

#: the linear Coriolis family (the modules carrying ``f v`` as a
#: ``linear=True`` term); the conserving modules below subclass them
#: for the ``f_coriolis`` declaration, so membership is always tested
#: through `carries_linear_rotation`
LINEAR_CORIOLIS = (FPlaneCoriolis, BetaPlaneCoriolis, RotationCoriolis)

#: the shallow-water velocity energy weight; the only ``metric_weight``
#: the correction knows how to subtract (it is the model's energy
#: metric — a Coriolis module weighted by anything else is not a
#: shallow-water rotation)
_WEIGHT = "csqr"

_CORE_HINT = "a shallow-water core, e.g. sw.DynamicalCore"
_F_HINT = ("a Coriolis module, e.g. sw.modules.FPlaneCoriolis — "
           "rotation is opt-in")


def _coord_names(coords: object) -> tuple[str, str]:
    """Validate the (zonal, meridional) coordinate names."""
    coords = tuple(coords)
    if (len(coords) != 2  # noqa: PLR2004 — zonal + meridional
            or not all(isinstance(c, str) for c in coords)
            or coords[0] == coords[1]):
        raise TypeError(
            "coords names the (zonal, meridional) coordinates: two "
            f"distinct strings, got {coords!r}")
    return coords


def carries_linear_rotation(module: object) -> bool:
    """
    Whether ``module`` carries the rotation as a ``linear=True`` term.

    Description
    -----------
    True for the linear Coriolis family (``FPlaneCoriolis`` and
    friends), False for the conserving modules of this file — which
    subclass them for the ``f_coriolis`` declaration but replace the
    term.

    Parameters
    ----------
    module : object
        Any module instance.

    Returns
    -------
    bool
        Whether the module contributes rotation to ``L``.
    """
    return (isinstance(module, LINEAR_CORIOLIS)
            and not isinstance(module, _ConservingRotation))


def check_rotation_modules(modules: object) -> None:
    """
    Refuse a module tuple that counts the rotation twice.

    Description
    -----------
    The one owner of the route-A / route-B combination rules, called
    from ``sw.Model`` (which sees the whole tuple before assembly) and
    from the conserving modules' ``bind`` (the backstop for explicit
    assembly). The rules:

    - a conserving (route B) module carries the rotation **whole**, so
      it may not be assembled with any other rotation module — a
      linear Coriolis module, the correction, or a second conserving
      one (through ``fr.model.Model`` directly, two ``f_coriolis``
      owners are already a ``FieldCollisionError``; this is the taught
      version);
    - the correction (route A) is a correction **to** the linear
      rotation, so it needs exactly one linear Coriolis module.

    Parameters
    ----------
    modules : Sequence
        The assembled module tuple (any order).

    Raises
    ------
    ValueError
        If the rotation is counted twice, or the correction has no
        (or more than one) linear Coriolis module to correct.
    """
    modules = tuple(modules)
    conserving = [module for module in modules
                  if isinstance(module, _ConservingRotation)]
    linear = [module for module in modules
              if carries_linear_rotation(module)]
    correction = [module for module in modules
                  if isinstance(module, CoriolisEnergyCorrection)]
    if conserving:
        others = [module for module in linear + correction + conserving
                  if module is not conserving[0]]
        if others:
            names = ", ".join(type(m).__name__ for m in others)
            raise ValueError(
                f"{type(conserving[0]).__name__} carries the rotation "
                "WHOLE (the conserving (f/h_bar)(h v)_bar), but this "
                f"model also assembles {names}: the rotation would be "
                "counted twice. Assemble either the conserving module "
                "ALONE (route B: exact energy, but the linear "
                "operator L has no rotation — no eigenmodes, "
                "projections, balance), or the linear Coriolis module "
                "plus the optional "
                "sw.modules.CoriolisEnergyCorrection (route A: the "
                "same tendency, and L unchanged)")
        return
    if not correction:
        return
    if not linear:
        raise ValueError(
            "CoriolisEnergyCorrection is a correction TO the linear "
            "Coriolis term, but this model carries no module "
            "contributing it (rotation is opt-in): assemble the "
            "correction alongside sw.modules.FPlaneCoriolis / "
            "BetaPlaneCoriolis / RotationCoriolis — a non-rotating "
            "model needs no correction")
    if len(linear) > 1:
        names = ", ".join(type(m).__name__ for m in linear)
        raise ValueError(
            "CoriolisEnergyCorrection corrects exactly one linear "
            f"Coriolis module, but this model carries {names}")


def _safe_pv_divide(
    numerator: ScalarField, thickness: ScalarField,
) -> ScalarField:
    r"""Return ``numerator / thickness`` with the divide VJP-sealed.

    Description
    -----------
    The conserving Coriolis ``f``-part of the potential vorticity,
    :math:`f / \bar h`, divides by the corner thickness ``h``. On a
    walled grid (the lat-lon sphere's polar caps, a closed basin) that
    thickness is an **exact zero** in the never-valid corner/halo
    padding, where the numerator vanishes too, so the bare quotient is a
    masked ``0/0``. The forward pass strips those cells (sealed/stripped
    before any output), but reverse-mode autodiff does not: the quotient
    VJP (:math:`-\mathrm{num}/h^2` with ``h == 0``) turns the zero
    cotangent of a sealed cell into ``0 * inf = NaN`` and poisons every
    gradient with a data path through the rotation — the same masked
    singularity ``SadournyAdvection._potential_vorticity`` cures for the
    advective PV divide (this is that same ``f``-part). Replacing the
    exact-zero denominators by 1 keeps the ratio finite there; valid
    cells (``h != 0``) divide by the true thickness and are bitwise
    unchanged, forward and reverse.
    """
    guarded = jnp.where(thickness.storage == 0.0, 1.0, thickness.storage)
    safe = ScalarField(
        thickness.grid, thickness.function_space, guarded,
        thickness.metadata, halo_valid=thickness.halo_valid)
    return numerator / safe


def conserving_rotation(
    state: object, *, coords: tuple[str, str], rossby: object,
    f_field: object = None,
) -> dict:
    r"""
    Return the exactly-conserving discrete Coriolis tendency.

    Description
    -----------
    The module docstring's ``(f / h_bar) (h v)_bar``: the ``f``-part
    of the vector-invariant PV flux, on the same NE vorticity corner
    and with the same thickness averages ``SadournyAdvection`` uses
    (flat and chart paths both). Carries **no** Rossby factor: the
    combined potential vorticity is :math:`(f + \mathrm{Ro}\,\zeta)/h`,
    whose :math:`\zeta` part is the (Ro-scaled) advection term and
    whose :math:`f` part — this one — is the unscaled rotation.

    Parameters
    ----------
    state : VectorField
        The model state; reads ``u``, ``v``, ``p``, ``csqr`` and
        ``f_coriolis``.
    coords : tuple[str, str]
        The (zonal, meridional) coordinate names.
    rossby : object
        The Rossby scaling (a traced ``ctx.params`` scalar), needed
        for the thickness ``h = c^2 + Ro p``.
    f_field : object, optional
        A **stage-time f(y) field** to use in place of the
        assembly-frozen ``f_coriolis`` field — the ramped beta-plane
        ``FieldBlend`` path (AR-D2 / R2): the conserving rotation
        consumes the fresh :math:`f(y,t) = f_0(t) + \beta(t)\,y`
        blend exactly like the frozen profile. ``None`` keeps the
        frozen field, so the static path is unchanged (default: None).

    Returns
    -------
    dict
        The ``u`` / ``v`` increments.
    """
    u, v, p = state["u"], state["v"], state["p"]
    c = state["csqr"]
    f = state["f_coriolis"] if f_field is None else f_field
    meridional = coords[1]

    # full geopotential thickness at the centre — the same h the
    # Sadourny scheme and the ekin_full diagnostic carry
    h = c.to(p) + rossby * p

    # the NE vorticity corner (identical construction to the Sadourny
    # scheme: it adopts each velocity's wall tag on the OTHER
    # velocity's axis, so the Dirichlet fill claims a zero wall value
    # — true, because every wall value consumed here carries the
    # wall-normal mass flux, an exact zero)
    corner = u.function_space.bare.replace(**{
        meridional: v.function_space.bare.factor(meridional)})

    if u.grid.chart_coords is None:
        f_u = (u * h.to(u)).to(corner)             # F^u, corner
        f_v = (v * h.to(v)).to(corner)             # F^v, corner
        # the f-part of q; the divide is VJP-sealed (h == 0 in the
        # walled/halo padding poisons the reverse pass — _safe_pv_divide)
        q_f = _safe_pv_divide(f.to(corner), h.to(corner))
        return {
            "u": (f_v * q_f).to(u),
            "v": -((f_u * q_f).to(v)),
        }
    return _conserving_chart(u, v, h, f, corner, coords)


def _conserving_chart(
    u: ScalarField,
    v: ScalarField,
    h: ScalarField,
    f: ScalarField,
    corner: object,
    coords: tuple[str, str],
) -> dict:
    r"""Return the metric-aware conserving Coriolis tendency.

    Description
    -----------
    The corner mass fluxes are the :math:`\sqrt g`-weighted fluxes
    :math:`F^i = \overline{\sqrt g\,\bar h\,u^i}` that the thickness
    divergence carries (``sqrt_g`` derived on each velocity's own
    space) — the placement that keeps the exchange antisymmetric under
    the metric thickness-weighted energy — and the covariant momentum
    tendencies are raised onto the prognostic contravariant
    components. Every metric factor sits exactly where
    ``SadournyAdvection._advect_chart`` puts it.
    """
    grid = u.grid
    zonal, meridional = coords
    sqg_u = grid.metric(u.function_space.bare, "sqrt_g")
    sqg_v = grid.metric(v.function_space.bare, "sqrt_g")
    f_u = (sqg_u * (u * h.to(u))).to(corner)       # F^lambda
    f_v = (sqg_v * (v * h.to(v))).to(corner)       # F^phi
    q_f = _safe_pv_divide(f.to(corner), h.to(corner))
    cov = Variance.COVARIANT
    t_u = ((f_v * q_f).to(u)).with_variance(cov)
    t_v = (-((f_u * q_f).to(v))).with_variance(cov)
    raise_index = grid.dispatch.resolve(
        "raise_index", t_u.function_space.bare)
    raised = raise_index(VectorField({
        zonal: t_u, meridional: t_v}))
    return {
        "u": raised[zonal].retag(u),
        "v": raised[meridional].retag(v),
    }


def linear_coriolis(
    state: object,
    *,
    coords: tuple[str, str],
    metric_weight: str | None,
) -> dict:
    """
    Return the linear Coriolis tendency of the installed module.

    Description
    -----------
    Delegates to the framework's own rotation expressions
    (``fr.model.modules.linear_rotation`` on a flat grid,
    ``chart_rotation`` on a chart grid) — the *same* functions the
    linear Coriolis modules call, so the correction below subtracts
    exactly what the linear module added, bit for bit.

    Parameters
    ----------
    state : VectorField
        The model state.
    coords : tuple[str, str]
        The (zonal, meridional) coordinate names (chart path).
    metric_weight : str | None
        The linear module's ``metric_weight`` (None: unweighted).

    Returns
    -------
    dict
        The ``u`` / ``v`` increments.
    """
    if state["u"].grid.chart_coords is None:
        return linear_rotation(state, metric_weight=metric_weight)
    return chart_rotation(state, coords=coords,
                          metric_weight=metric_weight)


# ================================================================
#  Route A: the optional energy-correction module
# ================================================================
class CoriolisEnergyCorrection(Module):

    r"""
    The optional exactly-conserving Coriolis correction (route A).

    Description
    -----------
    Carries the **difference** between the exactly-conserving discrete
    Coriolis term and the linear one,

    .. math::
        \frac{\overline f}{\overline h}\,\overline{h\,v} - f\,v ,

    as a single ``linear=False`` term, to be assembled **alongside** an
    ordinary linear Coriolis module (``sw.modules.FPlaneCoriolis`` /
    ``BetaPlaneCoriolis`` / ``RotationCoriolis``, which owns
    ``f_coriolis``). Their sum is the conserving form (module
    docstring), so the model's thickness-weighted energy
    (``sw.diagnostics.etot_full``) is conserved to machine precision
    instead of being produced at :math:`O(\mathrm{Ro})`.

    **The linear operator is untouched.** The term is declared
    nonlinear, so ``fr.model.linearize`` drops it and ``L`` is
    bit-for-bit the operator of a model without this module:
    eigenmodes, projections, optimal balance and IMEX-by-linearity are
    unaffected. The correction is *identically zero* on a rest state
    (both of its parts are proportional to the velocity), so it adds
    nothing to the linear model either.

    **Optional by construction.** Omitting the module is the off
    switch, and it reproduces today's behaviour bitwise. What running
    without it costs: the thickness-weighted energy is then conserved
    only to :math:`O(\mathrm{Ro})` — an error in the *invariant*, not
    in stability, and small by construction at small Rossby number.
    The **linear** model is unaffected either way. What it costs to
    switch on: one more nonlinear term in the tendency (and in the
    IMEX / term-filter bookkeeping).

    The paired linear module is found at bind: its ``metric_weight``
    is adopted (so the subtracted expression is exactly the one the
    linear module adds), and assembling this module without a linear
    Coriolis — or next to a conserving (route B) one, which already
    carries the whole term — is a taught error.

    Parameters
    ----------
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names in the grid's factor
        order — ``("lon", "lat")`` on the sphere chart
        (default: ``("x", "y")``).
    """

    field_references = (
        FieldReference("u", hint=_CORE_HINT),
        FieldReference("v", hint=_CORE_HINT),
        FieldReference("p", hint=_CORE_HINT),
        FieldReference("csqr", hint=_CORE_HINT),
        FieldReference("f_coriolis", hint=_F_HINT),
    )

    parameter_references = (Param(SCALING_ROSSBY, default=1.0),)

    def __init__(self, *, coords: tuple[str, str] = ("x", "y")) -> None:
        """Store the coordinate names; the weight is found at bind."""
        self._coords = _coord_names(coords)
        self._metric_weight: str | None = None

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    @property
    def metric_weight(self) -> str | None:
        """The paired linear module's weight field (set at bind)."""
        return self._metric_weight

    #: The corner chain reaches two cells per coordinate (centre ->
    #: corner -> velocity), exactly as in ``SadournyAdvection``; the
    #: term multiplies the traced ``scaling.rossby`` into the
    #: thickness, so the module declares its stencil width and is
    #: halo-trace exempt (V-N2).
    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (the corner chain)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    # ================================================================
    #  Bind: find the linear Coriolis module it corrects
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Adopt the paired linear Coriolis module's weight.

        Description
        -----------
        The correction must subtract *exactly* the expression the
        linear module adds, so its ``metric_weight`` is read from that
        module rather than re-declared here (a mismatch would silently
        break both the conservation and the route-A/route-B identity).
        The combination rules (`check_rotation_modules`) run first.

        Raises
        ------
        ValueError
            If the model carries no linear Coriolis module, more than
            one, a conserving (route B) Coriolis module (which already
            carries the whole term), or a linear module weighted by
            something other than ``csqr``.
        """
        modules = tuple(table.modules)
        check_rotation_modules(modules)
        linear = [module for module in modules
                  if carries_linear_rotation(module)]
        if getattr(linear[0], "_blend_active", False):
            raise ValueError(
                "CoriolisEnergyCorrection is paired with a Coriolis "
                "module carrying a ramped f(y) FieldBlend (a "
                "time-dependent f0/beta), but the correction subtracts "
                "the frozen f_coriolis snapshot, so route A would "
                "double-count the ramp and break the energy identity. "
                "For a ramped conserving run use route B "
                "(sw.modules.NonlinearBetaPlaneCoriolis(beta=Ramp(...))), "
                "which carries the whole conserving rotation on the "
                "stage-time blend; the route-A correction under a ramped "
                "f is a FieldBlend follow-up (roadmap 'Generalized "
                "adiabatic ramping')")
        weight = linear[0].metric_weight
        if weight not in (None, _WEIGHT):
            raise ValueError(
                f"the linear Coriolis module is weighted by "
                f"{weight!r}, but the shallow-water energy weight is "
                f"{_WEIGHT!r}: the correction can only subtract the "
                "rotation of a module weighted by the model's own "
                "energy metric (construct the Coriolis module with "
                "metric_weight='csqr', or without a weight)")
        self._metric_weight = weight

    # ================================================================
    #  The term (nonlinear: it never enters L)
    # ================================================================
    @term(advances=("u", "v"), linear=False, name="correction")
    def correction(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``(f / h_bar) (h v)_bar - f v`` (and its v-counterpart).

        Description
        -----------
        The conserving discrete Coriolis term minus the linear one the
        paired module already contributed; the sum of the two is the
        conserving form. Both parts are proportional to the velocity,
        so the correction is **exactly zero** on a rest state — it
        contributes nothing to the linearization (and it is declared
        ``linear=False`` regardless, so ``L`` never sees it).
        """
        rossby = ctx.params[SCALING_ROSSBY]
        full = conserving_rotation(state, coords=self._coords,
                                   rossby=rossby)
        linear = linear_coriolis(state, coords=self._coords,
                                 metric_weight=self._metric_weight)
        return {name: full[name] - linear[name]
                for name in ("u", "v")}


# ================================================================
#  Route B: the full nonlinear Coriolis modules
# ================================================================
class _ConservingRotation:

    r"""
    Route-B mixin: replace the linear rotation term by the whole one.

    Description
    -----------
    Reuses the linear family's ``f_coriolis`` declaration (and its
    bind-time grid validation) and overrides the ``coriolis`` term
    with the exactly-conserving ``(f / h_bar) (h v)_bar``. It drops
    the ``coriolis.f0`` provide of ``FPlaneCoriolis``: the constant is
    still the module's leaf, but publishing it would be a false claim
    that the linear operator carries that rotation, and the analytic
    eigenmode machinery reads the provide as exactly that claim.

    Takes no ``metric_weight``: the conserving form weights the
    rotation by the **thickness** itself, which is the exact energy
    weight for any depth profile (variable depth included).
    """

    #: the honesty gate: this module's rotation is NOT in L
    #: (``fr.model.require_linear_operator``)
    linear_operator_gap = (
        "it carries the rotation inside a nonlinear thickness ratio "
        "(f/h_bar)(h v)_bar, so the linear operator L has NO rotation "
        "at all — assemble the linear Coriolis module plus the "
        "optional sw.modules.CoriolisEnergyCorrection (route A) "
        "instead: same tendency, same exact energy conservation, and "
        "L unchanged")

    #: never provide ``coriolis.f0`` (provides-implies-constancy is
    #: read by the analytic eigenmodes as "L rotates at f0")
    parameter_declarations = ()

    parameter_references = (Param(SCALING_ROSSBY, default=1.0),)

    @property
    def metric_weight(self) -> None:
        """Always None: the thickness IS the weight (docstring)."""
        return None

    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v plus the thickness ingredients ``p`` and ``csqr``."""
        return (
            FieldReference("u", hint=_CORE_HINT),
            FieldReference("v", hint=_CORE_HINT),
            FieldReference("p", hint=_CORE_HINT),
            FieldReference("csqr", hint=_CORE_HINT),
        )

    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (the corner chain)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    def bind(self, table) -> None:  # noqa: ANN001
        """Refuse to co-exist with any other rotation module.

        Description
        -----------
        This module carries the rotation **whole**; a linear Coriolis
        module (or the route-A correction, or a second conserving
        module) next to it would count it twice
        (`check_rotation_modules`). Two ``f_coriolis`` owners are
        already a ``FieldCollisionError`` at assembly step 1, so the
        combination this catches at bind is the correction. The grid
        validation of the linear base class runs afterwards (chart /
        orthogonal checks).

        Raises
        ------
        ValueError
            If the model carries any other Coriolis module.
        """
        check_rotation_modules(table.modules)
        super().bind(table)

    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report nothing: the conserving rotation is ``linear=False``.

        A ramped ``f0``/``beta`` on a route-B module feeds the
        **nonlinear** ``(f/h_bar)(h v)_bar`` term (part of ``N``), not a
        ``linear=True`` term, so a frozen-``L`` stepper handles its time
        dependence correctly (AR-D7) — exactly as ``scaling.rossby``
        does. (Such a model is already refused by every ``L``-consumer
        through the ``linear_operator_gap``, but the honesty seam must
        still report nothing here.)
        """
        return ()

    @term(advances=("u", "v"), linear=False, name="coriolis")
    def coriolis(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``du = (f/h_bar) (h v)_bar``; ``dv = -(f/h_bar) (h u)_bar``.

        Description
        -----------
        The exactly-conserving discrete rotation, whole (module
        docstring): the ``f``-part of the vector-invariant PV flux, on
        the Sadourny corner, with the metric on a chart grid.
        Declared ``linear=False`` — it *is* nonlinear in ``h``, and
        the declaration is what keeps the model honest about ``L``.

        A ramped beta-plane (``NonlinearBetaPlaneCoriolis`` with a
        ``FieldBlend``-active ``f0``/``beta``) reads the fresh stage-time
        blend ``f(y,t)`` instead of the frozen field, so the conserving
        channel supports a ramped ``beta`` end to end (AR-D2 / R2). The
        static path (and the f-plane / chart route-B modules, which
        carry no field blend) leaves ``f_field`` ``None`` and reads the
        frozen ``f_coriolis`` unchanged.
        """
        stage_blend = getattr(self, "_stage_blend_f", None)
        f_field = (stage_blend(state, ctx)
                   if stage_blend is not None else None)
        return conserving_rotation(
            state, coords=self._coords,
            rossby=ctx.params[SCALING_ROSSBY], f_field=f_field)


class NonlinearFPlaneCoriolis(_ConservingRotation, FPlaneCoriolis):

    r"""
    Conserving f-plane Coriolis (route B): constant ``f``, whole term.

    Description
    -----------
    ``FPlaneCoriolis`` with the linear term replaced by the
    exactly-conserving ``(f / h_bar) (h v)_bar`` (module docstring):
    the thickness-weighted energy is conserved to machine precision.

    **The linear operator L loses rotation entirely.** With the
    rotation declared ``linear=False``, eigenmodes, projections,
    optimal balance and IMEX-by-linearity are **invalid — not
    degraded, wrong** — and the machinery refuses this model
    (``fr.model.require_linear_operator``); the module provides no
    ``coriolis.f0`` for the same reason. Use it for forward runs that
    never touch the linear machinery. If you need ``L``, assemble
    ``sw.modules.FPlaneCoriolis`` plus the optional
    :class:`CoriolisEnergyCorrection` (route A) — the same tendency,
    the same exact conservation, ``L`` unchanged.

    Parameters
    ----------
    f0 : float, optional
        The constant Coriolis parameter :math:`f_0` (default: 1.0).
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names (default:
        ``("x", "y")``).
    """

    def __init__(
        self, f0: float = 1.0, *,
        coords: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Store the Coriolis parameter and the coordinate names.

        Raises
        ------
        TypeError
            If ``f0`` is time-dependent: the conserving f-plane rotation
            carries no field blend (unlike the beta-plane), so its
            ``(f/h_bar)(h v)_bar`` term would silently read the frozen
            ``f0(0)`` snapshot. A ramped f-plane rotation belongs on the
            *linear* module (``sw.modules.FPlaneCoriolis(f0=Ramp(...))``,
            R1); a conserving ramped f0 is a ``FieldBlend`` follow-up.
        """
        if isinstance(f0, TimeDependent):
            raise TypeError(
                f"NonlinearFPlaneCoriolis f0={f0!r} is time-dependent, "
                "but the conserving f-plane rotation carries the whole "
                "(f/h_bar)(h v)_bar term with no field blend, so it "
                "would silently freeze f0 at t=0. Ramp f0 on the linear "
                "sw.modules.FPlaneCoriolis(f0=Ramp(...)) instead (R1), "
                "or use the conserving beta-plane "
                "sw.modules.NonlinearBetaPlaneCoriolis for a ramped "
                "rotation via its FieldBlend (roadmap 'Generalized "
                "adiabatic ramping')")
        FPlaneCoriolis.__init__(self, f0)
        self._coords = _coord_names(coords)


class NonlinearBetaPlaneCoriolis(_ConservingRotation, BetaPlaneCoriolis):

    r"""
    Conserving beta-plane Coriolis (route B): ``f(y)``, whole term.

    Description
    -----------
    ``BetaPlaneCoriolis`` with the linear term replaced by the
    exactly-conserving ``(f / h_bar) (h v)_bar`` (module docstring).
    **The linear operator L loses rotation entirely** — see
    :class:`NonlinearFPlaneCoriolis`; for an ``L``-carrying model use
    ``sw.modules.BetaPlaneCoriolis`` plus the optional
    :class:`CoriolisEnergyCorrection`.

    Parameters
    ----------
    f0 : float, optional
        Reference Coriolis parameter at ``y = 0`` (default: 1.0).
    beta : float, optional
        Meridional gradient :math:`\beta` (default: 0.0).
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names; the meridional one
        is the coordinate ``f`` varies along (default:
        ``("x", "y")``).
    """

    def __init__(
        self, f0: float = 1.0, beta: float = 0.0, *,
        coords: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Store the leaves and the coordinate names."""
        coords = _coord_names(coords)
        BetaPlaneCoriolis.__init__(self, f0, beta,
                                   meridional=coords[1])
        self._coords = coords


class NonlinearRotationCoriolis(_ConservingRotation, RotationCoriolis):

    r"""
    Conserving chart rotation (route B): ``f = 2 Omega . n``, whole.

    Description
    -----------
    ``RotationCoriolis`` (the chart-generic rotation, e.g. the lat-lon
    sphere) with the linear term replaced by the exactly-conserving
    ``(f / h_bar) (h v)_bar``, every average carrying the metric
    exactly as ``SadournyAdvection`` places it (module docstring). On
    the sphere this is what turns the 3.7e-2 relative energy
    production of the split Coriolis term into machine zero.

    **The linear operator L loses rotation entirely** — see
    :class:`NonlinearFPlaneCoriolis`; for an ``L``-carrying model use
    ``sw.modules.RotationCoriolis(metric_weight="csqr")`` plus the
    optional :class:`CoriolisEnergyCorrection`.

    Parameters
    ----------
    omega : tuple[float, float, float] | jax.Array, optional
        The **ambient** rotation vector (default: ``(0.0, 0.0, 1.0)``).
    coords : tuple[str, str], optional
        The chart coordinate names in the grid's factor order
        (default: ``("lon", "lat")``).
    """

    def __init__(
        self,
        omega: tuple[float, float, float] = (0.0, 0.0, 1.0),
        *,
        coords: tuple[str, str] = ("lon", "lat"),
    ) -> None:
        """Store the rotation vector and the chart coordinates."""
        RotationCoriolis.__init__(self, omega, coords=coords)
