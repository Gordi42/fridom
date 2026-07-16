r"""
Shared Coriolis modules: f-plane, beta-plane, chart rotation.

Description
-----------
The framework's reusable Coriolis module library (D2.1 module-library
sharing): both the nonhydrostatic and shallow-water ports consume
``fr.modules.FPlaneCoriolis`` / ``fr.modules.BetaPlaneCoriolis`` — one
clean, field-based implementation instead of a per-package copy.
``RotationCoriolis`` extends the family to chart-coupled grids
(coordinate-systems plan, stage C2): it takes the **ambient rotation
vector** :math:`\vec\Omega` and derives the Coriolis parameter from
the chart's own geometry, :math:`f = 2\,\vec\Omega\cdot\hat n` (see
its class docstring for the derivation). The lat-lon sphere with
:math:`\vec\Omega = (0, 0, \Omega)` is its special case, for which
the derived field is the familiar
:math:`f = 2\,\Omega\sin(\varphi)`.

**Rotation is opt-in.** The preset factories (``sw.Model``,
``nh.Model``) take ``coriolis=None`` — the argument omitted — to mean
**no Coriolis force at all**: no module, no ``f_coriolis`` field, no
rotation term, and no ``coriolis.f0`` provide. A rotating run names
its rotation explicitly. There is therefore no null "no-rotation"
module: omitting the argument *is* the no-rotation option.

**Metric-blindness is an error on chart grids**: ``FPlaneCoriolis``
and ``BetaPlaneCoriolis`` rotate *Cartesian* components with no
metric factors, so on a grid carrying an embedding chart they are
almost always wrong physics; they reject such a grid at bind with a
taught error naming ``RotationCoriolis``.

Following R2 (01_concepts D2.2), the Coriolis parameter is
*intrinsically spatial* — a constant on the f-plane, :math:`f(y)` on
the beta-plane — so it is carried as an AUXILIARY field ``f_coriolis``
on a **one-DOF** ``fr.Profile()`` (f-plane, constant everywhere) or a
meridional ``fr.Profile("y")`` (beta-plane, varying in y, broadcast in
x/z). Because the declared space is static, the f-plane and beta-plane
are two module *types*.

Each module also **carries** the linear rotation term

.. math::
    \partial_t u = f\,v , \qquad \partial_t v = -f\,u

written as **pure field arithmetic** in the energy-conserving
staggered form: ``f`` is sampled at the ``v`` faces and the ``u``
equation averages the flux (``(f_at_v * v).to(u)``), which keeps the
rotation exactly M-skew-adjoint for any ``f`` profile (see the term
docstring). Because ``f_coriolis`` is a *field*
(not a traced ``ctx.params`` scalar), the term needs no ``.with_data``
raw-scalar bypass and declares no ``extra_halo`` — the GAP-A
ConstantSpace/Profile broadcast makes the ``Profile -> nodal`` lift
trace cleanly and the ``.to`` interpolations are halo-traced normally.
The term is 2-D (``u``, ``v`` only) and identical for both the 3-D
nonhydrostatic and 2-D shallow-water cores; it never references ``w``
or the pressure/geopotential.

**Weighted velocity metrics** (``metric_weight``): when a model's
energy metric weights the velocities by a spatially varying field —
the variable-depth shallow water, whose conserved energy is
:math:`\tfrac12\int c^2(u^2 + v^2) + p^2` with :math:`c^2(y)` the
``csqr`` field — the unweighted staggered rotation is no longer
exactly M-skew: the ``u`` and ``v`` nodes sit at different ``y``
positions, so the weight cannot cancel across the interpolation.
Passing ``metric_weight="csqr"`` switches the term to the
**thickness-weighted flux form** (the linearized Sadourny/Arakawa
energy-conserving Coriolis)

.. math::
    \partial_t u = f\,\bar v , \qquad
    \partial_t v = -\overline{w\,f\,u}\,/\,w ,

which is exactly M-skew under ``diag(w, w, ...)`` for **any** ``f``
and any strictly positive weight profile, and reduces to the
unweighted form for a constant weight (up to float rounding). The
default (``None``) keeps the v1 form bit-for-bit.

**Provides-implies-constancy** (02_rules): ``FPlaneCoriolis`` provides
the constant ``coriolis.f0`` (analytic consumers such as
``eigenmodes.from_model`` rely on the provide as the constancy check);
``BetaPlaneCoriolis`` provides only ``coriolis.beta`` and must **not**
provide ``coriolis.f0`` (its ``f`` is the ``f(y)`` field, not a
scalar — an f0 provide would be a false constancy claim).
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.field_blend import BlendIngredient, FieldBlend
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import CORIOLIS_BETA, CORIOLIS_F0
from fridom.model.terms import term
from fridom.model.time_dependent import TimeDependent, resolve_at
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.space_patterns import Profile

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField

_U_HINT = ("velocities are declared by a dynamical-core module, "
           "e.g. nh.DynamicalCore or sw.DynamicalCore")


def linear_rotation(
    state: object,
    *,
    metric_weight: str | None = None,
    f_override: object = None,
    f_field: object = None,
) -> dict:
    r"""
    Return the linear staggered rotation ``f v`` / ``-f u`` (flat).

    Description
    -----------
    The **single source of truth** for the metric-blind (Cartesian)
    rotation expression: the term of ``FPlaneCoriolis`` /
    ``BetaPlaneCoriolis`` delegates here, and so does the
    shallow-water energy correction (which must subtract *exactly*
    this expression — see
    ``fridom.shallowwater2.modules.coriolis``), so the two can never
    drift apart. See the ``coriolis`` term docstring for the
    discrete-skewness argument.

    Parameters
    ----------
    state : VectorField
        The model state; reads ``u``, ``v``, ``f_coriolis`` and (when
        named) the metric-weight field.
    metric_weight : str | None, optional
        Name of the velocity energy-metric weight field ``w``;
        ``None`` is the unweighted form (default: None).
    f_override : object, optional
        A **constant-in-space** stage-time scalar to use in place of
        the ``f_coriolis`` field — the f-plane time-dependent path
        (AR-D7 / R1): when a ``TimeDependent`` ``f0`` drives the
        rotation, the term reads ``f0(t)`` at stage time rather than
        the assembly-frozen field. Already constant in space, so the
        field lift ``.to(u)`` is skipped. ``None`` keeps the field
        path, so the static (plain-float) case is bit-identical
        (default: None).
    f_field : object, optional
        A **stage-time f(y) field** (a ``ScalarField`` on the
        ``f_coriolis`` space) to use in place of the assembly-frozen
        ``f_coriolis`` field — the beta-plane ``FieldBlend`` path (AR-D2
        / R2): the ramped ``f(y,t) = f0(t) + beta(t)*y`` blended from
        the ingredient profiles at stage time, lifted onto the ``u``
        faces exactly like the frozen field. ``None`` keeps the frozen
        field (default: None).

    Returns
    -------
    dict
        The ``u`` / ``v`` increments.
    """
    u, v = state["u"], state["v"]
    # a scalar override is already constant in space (the f-plane), so
    # the field lift ``.to(u)`` is a no-op broadcast and is skipped; a
    # blended f(y) field (beta-plane) is lifted like the frozen field
    if f_override is not None:
        f_u = f_override
    elif f_field is not None:
        f_u = f_field.to(u)
    else:
        f_u = state["f_coriolis"].to(u)
    if metric_weight is None:
        return {
            "u": f_u * v.to(u),
            "v": -((f_u * u).to(v)),
        }
    w = state[metric_weight]
    return {
        "u": f_u * v.to(u),
        "v": -((w.to(u) * f_u * u).to(v)) / w.to(v),
    }


def chart_rotation(
    state: object,
    *,
    coords: tuple[str, str],
    metric_weight: str | None = None,
) -> dict:
    r"""
    Return the metric-aware rotation of ``RotationCoriolis`` (chart).

    Description
    -----------
    The **single source of truth** for the chart rotation expression
    (``RotationCoriolis.coriolis`` delegates here; the shallow-water
    energy correction subtracts exactly this). The class docstring's
    energy-conserving flux form: the flux weight
    :math:`G = f\,g\,w` is sampled once at the ``u`` faces and
    averaged back to the ``v`` faces *inside* the flux, so the pair is
    exactly M-skew under :math:`\mathrm{diag}(W_1, W_2)`.

    Parameters
    ----------
    state : VectorField
        The model state; reads ``u``, ``v``, ``f_coriolis`` and (when
        named) the metric-weight field.
    coords : tuple[str, str]
        The chart coordinate names, in the grid's factor order.
    metric_weight : str | None, optional
        Name of the velocity energy-metric weight field ``w``
        (default: None).

    Returns
    -------
    dict
        The ``u`` / ``v`` increments (contravariant components).
    """
    u, v, f = state["u"], state["v"], state["f_coriolis"]
    grid = u.grid
    c_1, c_2 = coords
    u_space = u.function_space.bare
    v_space = v.function_space.bare
    sqg_u = grid.metric(u_space, "sqrt_g")
    sqg_v = grid.metric(v_space, "sqrt_g")
    g_uu = grid.metric(u_space, f"g_{c_1}{c_1}")
    g_vv = grid.metric(v_space, f"g_{c_2}{c_2}")
    f_u = f.to(u)
    flux_weight = f_u * (sqg_u * sqg_u)          # G = f g (w below)
    w_1 = sqg_u * g_uu
    w_2 = sqg_v * g_vv
    if metric_weight is not None:
        w = state[metric_weight]
        flux_weight = flux_weight * w.to(u)
        w_1 = w_1 * w.to(u)
        w_2 = w_2 * w.to(v)
    return {
        "u": flux_weight * v.to(u) / w_1,
        "v": -((flux_weight * u).to(v)) / w_2,
    }


@term(advances=("u", "v"), linear=True, name="coriolis")
def _coriolis(self, state, ctx) -> dict:  # noqa: ANN001
    r"""``du/dt = f v``; ``dv/dt = -f u`` as pure field arithmetic.

    The **energy-conserving** staggered form of the v1 framework:
    ``f`` is sampled once, at the ``u`` faces, and the ``v`` equation
    averages the *flux* ``(f u)`` back to the ``v`` faces. Because
    the ``.to`` interpolations between the ``u`` and ``v`` spaces are
    measure-weighted adjoints of each other, the pair
    ``u += f_u (v.to(u))``, ``v -= (f_u u).to(v)`` is exactly
    M-skew-adjoint for **any** ``f`` profile (rotation does no
    work); sampling ``f`` per target face instead would break
    discrete energy conservation for a varying ``f``. For a constant
    ``f`` the two forms coincide bit-for-bit (``.to`` is linear).
    Shared verbatim by the f-plane (constant ``f``) and beta-plane
    (``f(y)``) module types — the only difference between them is
    the *space* of ``f_coriolis``, not the coupling.

    With a ``metric_weight`` field ``w`` (a varying velocity energy
    weight, e.g. the variable-depth ``csqr``) the interpolated flux
    is thickness-weighted, ``v -= (w_u f_u u).to(v) / w_v``: under
    ``diag(w, w, ...)`` the M-adjoint of ``u += f_u (v.to(u))`` is
    exactly that expression (the weight enters the flux at the
    ``u`` nodes and cancels at the ``v`` nodes), so the pair stays
    M-skew for any ``f`` and any positive ``w`` profile — the
    linearized Sadourny/Arakawa pairing. For a constant ``w`` it
    coincides with the unweighted form mathematically (``.to`` is
    linear); in floating point the two round differently by up to
    1 ulp (bitwise-identical only for power-of-two ``w``), so the
    weighted form is the safe default whenever a weight field
    exists.

    The expression itself lives in `linear_rotation` (one owner: the
    shallow-water energy-correction module subtracts exactly this).

    When the module carries a **time-dependent** constant-in-space
    ``f0`` (the f-plane R1 path), ``_stage_scalar_f`` returns the
    stage-time value ``f0(t)`` and the term reads it instead of the
    assembly-frozen field. When it carries a ramped ``f(y,t)`` (the
    beta-plane ``FieldBlend`` path, R2), ``_stage_blend_f`` returns the
    stage-time blended field. A fully static module (plain-float ``f0``
    and ``beta``) leaves both overrides ``None`` and the assembly-frozen
    field path runs bit-identically.
    """
    stage_f = getattr(self, "_stage_scalar_f", None)
    f_override = stage_f(ctx) if stage_f is not None else None
    stage_blend = getattr(self, "_stage_blend_f", None)
    f_field = stage_blend(state, ctx) if stage_blend is not None else None
    return linear_rotation(
        state, metric_weight=self._metric_weight,
        f_override=f_override, f_field=f_field)


_WEIGHT_HINT = ("the velocity energy-metric weight field (e.g. the "
                "shallow-water csqr, declared by its dynamical core)")


def _rotation_references(
    metric_weight: str | None,
) -> tuple[FieldReference, ...]:
    """Build the u/v references, plus the weight field if named."""
    refs = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    if metric_weight is not None:
        refs += (FieldReference(metric_weight, hint=_WEIGHT_HINT),)
    return refs


def _chart_coord_names(coords: object) -> tuple[str, str]:
    """Validate the (first, second) chart coordinate names."""
    coords = tuple(coords)
    if (len(coords) != 2  # noqa: PLR2004 — a surface chart
            or not all(isinstance(c, str) for c in coords)
            or coords[0] == coords[1]):
        raise TypeError(
            "coords names the (zonal, meridional) chart "
            f"coordinates: two distinct strings, got {coords!r}")
    return coords


def _rotation_vector(omega: object) -> jnp.ndarray:
    """Validate the ambient rotation vector (three components)."""
    vector = jnp.asarray(omega, dtype=dtype_real())
    if vector.shape != (3,):
        raise TypeError(
            "omega is the AMBIENT rotation vector of the chart's "
            "embedding space: three components, e.g. "
            f"omega=(0.0, 0.0, 7.292e-5); got {omega!r}")
    return vector


def _f_const_ingredient(
    self: object, grid: object, space: object,  # noqa: ARG001
) -> ScalarField:
    """Owner-method default: the constant unit profile (all ones).

    The ``f0``-weighted ingredient of the beta-plane ``FieldBlend``:
    ``f(y,t) = f0(t) * 1 + beta(t) * y``. Materialized once at assembly
    on the meridional profile; the pointwise blend scales it by the
    stage-time ``f0(t)``.
    """
    return grid.create_field(
        space, data=jnp.ones(space.shape), name="f_coriolis_const")


def _f_grad_ingredient(
    self: object, grid: object, space: object,
) -> ScalarField:
    """Owner-method default: the meridional coordinate profile ``y``.

    The ``beta``-weighted ingredient of the beta-plane ``FieldBlend``;
    the ``init`` signature is stamped dynamically to name the module's
    own meridional coordinate (mirrors ``BetaPlaneCoriolis._f_default``).
    Materialized once at assembly; the pointwise blend scales it by the
    stage-time ``beta(t)``.
    """
    mer = self._meridional

    def init(**coords: object) -> object:
        return coords[mer]

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(
            mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return grid.create_field(space, init=init, name="f_coriolis_grad")


#: the beta-plane Coriolis blend ``f(y,t) = f0(t)*1 + beta(t)*y`` — two
#: assembly-materialized profiles (the constant unit and the meridional
#: coordinate) weighted by the module's own ``f0`` / ``beta`` leaves,
#: read at stage time (AR-D2). The two-endpoint paper form
#: ``f0 + rho(t/tau)*beta*y`` is the special case ``f0`` static, ``beta``
#: a ``Ramp`` (its ``0 -> beta`` ramp is ``rho(t/tau)*beta``).
_BETA_BLEND = FieldBlend((
    BlendIngredient("f_coriolis_const", weight="f0",
                    build=_f_const_ingredient),
    BlendIngredient("f_coriolis_grad", weight="beta",
                    build=_f_grad_ingredient),
))


def _reject_chart_grid(module: Module, table: object) -> None:
    """Refuse a metric-blind rotation on a chart-coupled grid.

    The f-plane and beta-plane terms rotate the velocity components
    as if they were Cartesian (no ``sqrt(g)``, no ``g_ij``): on a
    chart grid the prognostic velocities are *contravariant*
    components, so the term would be silently wrong physics (and
    would do work against the metric energy). Better a taught error
    than a plausible-looking wrong answer.
    """
    chart = table.grid.chart_coords
    if chart is None:
        return
    raise ValueError(
        f"{type(module).__name__} is metric-blind (it rotates "
        "Cartesian velocity components), but this grid carries an "
        f"embedding chart on {chart}, whose velocities are "
        "contravariant components: use "
        "fr.modules.RotationCoriolis(omega=(0.0, 0.0, Omega), "
        f"coords={chart!r}) — it derives f = 2 Omega . n_hat from "
        "the chart itself, and the lat-lon sphere with a polar Omega "
        "gives f = 2 Omega sin(lat) — or omit coriolis= entirely to "
        "run without rotation")


@partial(jaxify, dynamic=("f0",))
class FPlaneCoriolis(Module):

    r"""
    Constant-rotation Coriolis on the f-plane; provides ``coriolis.f0``.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on ``fr.Profile()``
    (one degree of freedom, :math:`f \equiv f_0` everywhere) and
    carries the linear rotation term. Provides the constant
    ``coriolis.f0`` (provides-implies-constancy).

    Parameters
    ----------
    f0 : float, optional
        The constant Coriolis parameter :math:`f_0` (default: 1.0).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``); switches
        the rotation to the thickness-weighted flux form, the
        M-skew pairing under ``diag(w, w, ...)`` (default: None).
    """

    def __init__(
        self, f0: float = 1.0, *, metric_weight: str | None = None,
    ) -> None:
        """Store the Coriolis parameter as a dynamic leaf."""
        self.f0 = leaf(f0)
        self._metric_weight = metric_weight

    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_F0, attr="f0", units="1/s",
                             doc="constant Coriolis parameter"),
    )

    @property
    def metric_weight(self) -> str | None:
        """The velocity energy-metric weight field name (or None)."""
        return self._metric_weight

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The one-DOF constant Coriolis field (``fr.Profile()``)."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._f_default,
                long_name="Coriolis parameter", units="1/s"),
        )

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: fill the profile with ``f0``.

        A time-dependent ``f0`` (an ``fr.Ramp``) is materialized at
        ``t = 0`` (``resolve_at(self.f0, 0.0)``) so the AUXILIARY
        field keeps a valid static treedef; the rotation term then
        overrides it with the stage-time value ``f0(t)`` per step
        (``_stage_scalar_f``), so the frozen field value is never read
        on the time-dependent path. A plain-float ``f0`` is
        ``resolve_at``-identity, so this line is bit-identical to the
        static case.

        No ``grid.sync`` pre-syncing: the GAP-B fix records the
        consumption-side exchange in an external identity cache, so a
        carry-resident AUXILIARY field keeps a stable scan treedef
        without being pre-synced to full halo.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, resolve_at(self.f0, 0.0)),
            name="f_coriolis")

    def _stage_scalar_f(self, ctx: object) -> object | None:
        """Return the stage-time scalar ``f0(t)`` when ``f0`` is ramped.

        Description
        -----------
        The f-plane ``f`` is constant in space, so a time-dependent
        ``f0`` needs no field rewrite (AR-D2 is R2): the rotation term
        reads the provided ``coriolis.f0`` from ``ctx.params``, which
        the binding table has already resolved at the stage clock time
        (``eval_params`` applies ``resolve_at`` — the same seam the
        ``scaling.rossby`` ramp rides), so the value is correct in the
        assembly dry run, ``model.tendency`` and every stepper stage
        alike. Returns ``None`` for a plain-float ``f0`` (the field
        path stays bit-identical); the static branch never touches
        ``ctx`` (host-side dispatch on the leaf type), so the term is
        still callable with ``ctx=None``.
        """
        if isinstance(self.f0, TimeDependent):
            return ctx.params[CORIOLIS_F0]
        return None

    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report a ramped ``coriolis.f0`` feeding the linear rotation.

        A time-dependent ``f0`` lives inside this module's
        ``linear=True`` rotation term, so a frozen-``L`` (exponential)
        stepper must refuse it (AR-D7); a plain-float ``f0`` reports
        nothing.
        """
        if isinstance(self.f0, TimeDependent):
            return (str(CORIOLIS_F0),)
        return ()

    def bind(self, table) -> None:  # noqa: ANN001
        """Reject chart-coupled grids (metric-blind rotation).

        Raises
        ------
        ValueError
            If the grid carries an embedding chart.
        """
        _reject_chart_grid(self, table)

    #: ``du/dt = f v``; ``dv/dt = -f u`` (shared rotation term).
    coriolis = _coriolis


@partial(jaxify, dynamic=("f0", "beta"))
class BetaPlaneCoriolis(Module):

    r"""
    Beta-plane Coriolis :math:`f(y) = f_0 + \beta y`.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on
    ``fr.Profile("y")`` (varies in the meridional coordinate,
    broadcast elsewhere) and carries the linear rotation term.
    Provides ``coriolis.beta`` and deliberately does **not** provide
    ``coriolis.f0`` (its Coriolis parameter is the ``f(y)`` field, not
    a constant — 02_rules).

    A time-dependent ``f0`` and/or ``beta`` (an ``fr.Ramp``, or a
    ``Ramp``-valued ``updates=`` on ``coriolis.f0`` / ``coriolis.beta``)
    drives the adiabatic-ramping ``FieldBlend`` (AR-D2 / R2):
    :math:`f(y,t) = f_0(t)\cdot 1 + \beta(t)\,y` is blended at stage
    time from two assembly-materialized profiles (the constant unit
    field and the meridional coordinate), so ramp-endpoint sweeps never
    recompile and the pointwise blend adds no halo traffic. The state
    then also carries the two ingredient fields ``f_coriolis_const`` /
    ``f_coriolis_grad``, and ``f_coriolis`` is materialized as the
    ``t = 0`` snapshot; the rotation term reads the fresh blend.

    Parameters
    ----------
    f0 : float | fr.Ramp, optional
        Reference Coriolis parameter at ``y = 0`` (default: 1.0).
    beta : float | fr.Ramp, optional
        Meridional gradient :math:`\beta = \mathrm{d}f/\mathrm{d}y`
        (default: 0.0).
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``); switches
        the rotation to the thickness-weighted flux form, the
        M-skew pairing under ``diag(w, w, ...)`` (default: None).
    """

    def __init__(
        self, f0: float = 1.0, beta: float = 0.0,
        *, meridional: str = "y", metric_weight: str | None = None,
    ) -> None:
        r"""Store the leaves and the meridional coordinate name.

        A time-dependent ``f0`` and/or ``beta`` (an ``fr.Ramp``) is
        supported through the beta-plane ``FieldBlend`` (AR-D2 / R2):
        the Coriolis parameter is the spatially varying field
        :math:`f(y,t) = f_0(t) + \beta(t)\,y`, blended at stage time
        from the assembly-materialized unit and meridional-coordinate
        profiles (see :data:`_BETA_BLEND`). The static (plain-float)
        path is untouched.
        """
        self.f0 = leaf(f0)
        self.beta = leaf(beta)
        self._meridional = meridional
        self._metric_weight = metric_weight

    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_BETA, attr="beta",
                             units="1/(m s)",
                             doc="meridional Coriolis gradient"),
    )

    @property
    def metric_weight(self) -> str | None:
        """The velocity energy-metric weight field name (or None)."""
        return self._metric_weight

    @property
    def _blend_active(self) -> bool:
        """Whether a ramped ``f0``/``beta`` drives the ``FieldBlend``.

        Computed from the *current* leaves rather than cached in
        ``__init__``: ``model.variant(updates=...)`` seeds a
        ``Ramp``-valued ``coriolis.beta`` via ``object.__setattr__``
        (bypassing ``__init__``), so the ramping ``Propagator`` legs
        must see a freshly re-evaluated predicate at re-assembly.
        Reads only the leaf *types* (structural, host-side).
        """
        return _BETA_BLEND.is_active(self)

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The ``f(y)`` field, plus the blend ingredients when ramped.

        The static (plain-float) path declares the single ``f_coriolis``
        profile exactly as before. A ramped ``f0``/``beta`` additionally
        declares the two ``FieldBlend`` ingredient profiles
        (``f_coriolis_const``, ``f_coriolis_grad``); ``f_coriolis``
        itself stays declared as the ``t = 0`` snapshot (so downstream
        consumers and I/O keep a valid field), while the rotation term
        reads the fresh stage-time blend.
        """
        f_coriolis = FieldDeclaration(
            "f_coriolis", space=Profile(self._meridional),
            lifecycle=Lifecycle.AUXILIARY, default=self._f_default,
            long_name="Coriolis parameter", units="1/s")
        if not self._blend_active:
            return (f_coriolis,)
        return (f_coriolis, *_BETA_BLEND.field_declarations(
            space=Profile(self._meridional),
            long_name="Coriolis parameter blend ingredient",
            units="1/s"))

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: materialize ``f0 + beta*y`` at ``t=0``.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``.
        A time-dependent ``f0``/``beta`` (an ``fr.Ramp``) is resolved at
        ``t = 0`` (``resolve_at``) so the AUXILIARY field keeps a valid
        static treedef; the rotation term then reads the stage-time
        blend, so this frozen value is never used on the ramped path. A
        plain-float ``f0``/``beta`` is ``resolve_at``-identity, so this
        line is bit-identical to the static case. No pre-syncing
        (GAP-B) — see ``FPlaneCoriolis._f_default``.
        """
        f0 = resolve_at(self.f0, 0.0)
        beta = resolve_at(self.beta, 0.0)
        mer = self._meridional

        def init(**coords: object) -> object:
            return f0 + beta * coords[mer]

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="f_coriolis")

    def _stage_blend_f(self, state: object, ctx: object) -> object | None:
        r"""Return the stage-time blended ``f(y,t)`` when ramped (AR-D2).

        Description
        -----------
        The beta-plane counterpart of ``FPlaneCoriolis._stage_scalar_f``:
        a ramped ``f0``/``beta`` makes ``f`` a field-valued blend
        :math:`f(y,t) = f_0(t) + \beta(t)\,y`, evaluated from the
        assembly-materialized ingredient profiles and the module's own
        leaves resolved at the stage clock time (``ctx.clock.time`` in a
        run, the bare dry-run/tendency scalar otherwise — the
        ``eval_params``-consistent seam). Returns ``None`` on the static
        (plain-float) path (host-side dispatch on the leaf types), so the
        rotation term reads the frozen field bit-identically and the
        branch never dereferences ``ctx`` there.
        """
        if not self._blend_active:
            return None
        time = getattr(ctx.clock, "time", ctx.clock)
        return _BETA_BLEND.evaluate(self, state, time)

    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report a ramped ``f0``/``beta`` feeding the linear rotation.

        A time-dependent ``f0`` or ``beta`` lives inside this module's
        ``linear=True`` rotation term, so a frozen-``L`` (exponential)
        stepper must refuse it (AR-D7); a fully plain-float module
        reports nothing.
        """
        names: list[str] = []
        if isinstance(self.f0, TimeDependent):
            names.append(str(CORIOLIS_F0))
        if isinstance(self.beta, TimeDependent):
            names.append(str(CORIOLIS_BETA))
        return tuple(names)

    def bind(self, table) -> None:  # noqa: ANN001
        """Reject chart-coupled grids (metric-blind rotation).

        Raises
        ------
        ValueError
            If the grid carries an embedding chart.
        """
        _reject_chart_grid(self, table)

    #: ``du/dt = f(y) v``; ``dv/dt = -f(y) u`` (shared rotation term).
    coriolis = _coriolis


@partial(jaxify, dynamic=("omega",))
class RotationCoriolis(Module):

    r"""
    Chart-generic rotation: :math:`f = 2\,\vec\Omega\cdot\hat n`.

    Description
    -----------
    The Coriolis acceleration is :math:`-2\,\vec\Omega\times\vec u`.
    For a fluid confined to a 2-D manifold embedded in
    :math:`\mathbb{R}^3` — every chart grid (coordinate-systems
    plan, stage C2) — split the ambient rotation vector at each
    point into its surface-normal and tangential parts,
    :math:`\vec\Omega = (\vec\Omega\cdot\hat n)\,\hat n +
    \vec\Omega_t`. With :math:`\vec u` tangent to the surface,
    :math:`\vec\Omega_t\times\vec u` is purely **normal** (a cross
    product of two tangent vectors) and is removed by the tangential
    projection — it is balanced by the constraint force that holds
    the fluid on the surface. What survives is a *local* rotation
    about the normal,

    .. math::
        \left(-2\,\vec\Omega\times\vec u\right)_{\rm tangential}
            = -f\,(\hat n\times\vec u) , \qquad
        f = 2\,\vec\Omega\cdot\hat n ,

    with no sphere-specific assumption: **f is a derived scalar
    field of the chart**, not a user formula. :math:`\hat n =
    (X_1\times X_2)/|X_1\times X_2|` comes from the chart's own
    tangent vectors (the ``normal_x``/``normal_y``/``normal_z``
    metrics of ``CoordinateMapping``).

    :math:`\hat n\times\vec u` is the 90-degree rotation in the
    tangent plane. In **contravariant** components (the chart-grid
    velocity convention: :math:`u^i = \dot u^i`), using
    :math:`(X_1\times X_2)\times X_1 = g_{11} X_2 - g_{12} X_1` and
    its partner, and :math:`|X_1\times X_2| = \sqrt g`:

    .. math::
        (\hat n\times\vec u)^1 = -\frac{u_2}{\sqrt g} , \qquad
        (\hat n\times\vec u)^2 = +\frac{u_1}{\sqrt g} , \qquad
        u_i = g_{ij}u^j ,

    so the tendency :math:`\partial_t u^i = -f\,(\hat n\times\vec
    u)^i` reads

    .. math::
        \partial_t u^1 = +\frac{f}{\sqrt g}
            \left(g_{12}u^1 + g_{22}u^2\right) , \qquad
        \partial_t u^2 = -\frac{f}{\sqrt g}
            \left(g_{11}u^1 + g_{12}u^2\right) .

    **Orthogonal charts only.** For a diagonal metric the cross
    terms drop and the tendency is
    :math:`\partial_t u^1 = f\,g_{22}u^2/\sqrt g =
    f\sqrt g\,g^{11}u^2`, :math:`\partial_t u^2 = -f\sqrt g\,
    g^{22}u^1` — exactly the (already validated) spherical form. The
    off-diagonal terms cannot be discretized skew-symmetrically on a
    staggered C-grid: the exact energy
    :math:`\tfrac12\int\sqrt g\,w\,g_{ij}u^iu^j` then carries a
    cross term :math:`g_{12}u^1u^2` whose two factors live at
    *different* nodes, so no local pairing reproduces it and the
    rotation would leak energy at O(1). Skew-symmetry is the
    load-bearing property here, so a non-orthogonal chart is a taught
    error at bind (this is the same condition the chart's
    ``orthogonal=True`` declaration asserts to seed the grid's
    ``RaiseIndex(diagonal=True)`` / ``LowerIndex(diagonal=True)``).

    **Exact discrete skew-symmetry.** With the per-point flux weight
    :math:`G = f\,g\,w` (:math:`g = \det g_{ij} = (\sqrt g)^2`,
    sampled once at the ``u`` faces) and the energy weights
    :math:`W_1 = \sqrt g\,g_{11}\,w`, :math:`W_2 = \sqrt g\,
    g_{22}\,w`,

    .. math::
        \partial_t u^1 = \frac{G\,\overline{u^2}}{W_1} , \qquad
        \partial_t u^2 = -\,\frac{\overline{G\,u^1}}{W_2} ,

    the pair is exactly M-skew-adjoint under
    :math:`\mathrm{diag}(W_1, W_2)` for **any** ``f`` and any
    positive weight: the measure-weighted ``.to`` averages are
    adjoints of each other, and the :math:`W` factors cancel against
    the energy metric, so
    :math:`\sum W_1 u^1\,\partial_t u^1 + \sum W_2 u^2\,\partial_t
    u^2 = \sum (G u^1)\,\overline{u^2} - \sum \overline{(G u^1)}\,
    u^2 = 0` (rotation does no work, to the rounding of the
    pointwise :math:`W\,(G/W)` round-trip). This is why the metric
    factors are sampled *inside* the flux rather than per target
    face. The metric factors are derived per application via
    ``grid.metric`` and never cached (rules 2.3/3.8).

    Sanity checks (all covered by tests): the lat-lon sphere chart
    with :math:`\vec\Omega = (0,0,\Omega)` gives
    :math:`f = 2\Omega\sin\varphi` — **the classical spherical
    Coriolis parameter is a derived special case**; the
    flat identity chart :math:`X = (x, y, 0)` gives
    :math:`f = 2\Omega` — **the f-plane is a derived special case**,
    bitwise equal to ``FPlaneCoriolis(f0=2*Omega)``; a torus chart
    gives the :math:`f` of its analytic normal.

    Parameters
    ----------
    omega : tuple[float, float, float] | jax.Array, optional
        The **ambient** rotation vector :math:`\vec\Omega` in the
        chart's embedding coordinates, e.g. ``(0.0, 0.0, 7.292e-5)``
        for Earth with the lat-lon chart (default:
        ``(0.0, 0.0, 1.0)``).
    coords : tuple[str, str], optional
        The chart coordinate names in the grid's factor order; the
        orientation of :math:`\hat n` follows their order
        (default: ``("lon", "lat")``).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``)
        (default: None).
    """

    #: relative size of ``g_12`` tolerated as rounding at bind
    _ORTHOGONAL_TOL = 1e-10

    def __init__(
        self,
        omega: tuple[float, float, float] = (0.0, 0.0, 1.0),
        *,
        coords: tuple[str, str] = ("lon", "lat"),
        metric_weight: str | None = None,
    ) -> None:
        """Store the rotation vector and the coordinate names."""
        self.omega = leaf(_rotation_vector(omega))
        self._coords: tuple[str, str] = _chart_coord_names(coords)
        self._metric_weight = metric_weight

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coords(self) -> tuple[str, str]:
        """The chart coordinate names, in the grid's factor order."""
        return self._coords

    @property
    def metric_weight(self) -> str | None:
        """The velocity energy-metric weight field name (or None)."""
        return self._metric_weight

    # ================================================================
    #  Declarations
    # ================================================================
    #: two interpolation hops per axis at most (v -> u and the flux
    #: back); the metric fields multiply a traced flux (raw metric
    #: data the halo tracer cannot follow), so the module declares
    #: its stencil width and is halo-trace exempt (V-N2)
    @property
    def extra_halo(self) -> HaloSpec:
        """One halo cell per chart axis (the ``.to`` averages)."""
        return HaloSpec(dict.fromkeys(self._coords, 1))

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        r"""``f_coriolis`` on the chart's own two coordinates.

        The derived :math:`f = 2\,\vec\Omega\cdot\hat n` varies
        along **both** chart coordinates in general (it does not on
        the sphere or the torus with a polar :math:`\vec\Omega`,
        but it does for a tilted one), so the declared space is the
        full chart profile; the interpolations in the term are
        exact no-ops along a coordinate ``f`` is constant in.
        """
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(*self._coords),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._f_default,
                long_name="Coriolis parameter", units="1/s"),
        )

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: ``f = 2 Omega . n_hat``.

        The surface normal is a chart-derived metric of the grid's
        ``CoordinateMapping`` (one owner of the derivation, rules
        3.8), so the Coriolis parameter is assembled here as pure
        field arithmetic on the ``normal_<x|y|z>`` metrics. No
        pre-syncing (GAP-B).
        """
        omega = self.omega
        normal = [grid.metric(space, f"normal_{c}")
                  for c in ("x", "y", "z")]
        f = 2.0 * (omega[0] * normal[0] + omega[1] * normal[1]
                   + omega[2] * normal[2])
        return grid.create_field(
            space, data=jnp.broadcast_to(f.data, space.shape),
            name="f_coriolis")

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Require an **orthogonal** chart grid matching ``coords``.

        Raises
        ------
        ValueError
            If the grid carries no embedding chart, if the chart
            coordinate family does not match ``coords`` in the
            grid's factor order, or if the induced metric is not
            diagonal (see the class docstring: the off-diagonal
            rotation has no skew-symmetric staggered form).
        """
        name = type(self).__name__
        grid = table.grid
        chart = grid.chart_coords
        if chart is None:
            raise ValueError(
                f"{name} needs a chart-coupled grid (a "
                "CoordinateMapping with an embedding chart, e.g. "
                "the lat-lon sphere); on flat Cartesian grids use "
                "FPlaneCoriolis or BetaPlaneCoriolis")
        expected = tuple(
            member for member in grid.names if member in set(chart))
        if self._coords != expected:
            raise ValueError(
                f"{name} coords={self._coords!r} do not "
                f"match the grid's chart coordinates {expected!r} "
                "(in factor order); pass coords matching the grid")
        self._require_orthogonal(grid, table["u"].space)

    def _require_orthogonal(self, grid: object,
                            space: object) -> None:
        """Reject a chart whose induced metric has cross terms."""
        c_1, c_2 = self._coords
        off = jnp.abs(grid.metric(space, f"g_{c_1}{c_2}").data)
        scale = jnp.sqrt(grid.metric(space, f"g_{c_1}{c_1}").data
                         * grid.metric(space, f"g_{c_2}{c_2}").data)
        ratio = float(jnp.max(off / scale))
        if ratio > self._ORTHOGONAL_TOL:
            raise ValueError(
                f"{type(self).__name__} needs an ORTHOGONAL chart "
                f"(diagonal induced metric), but g_{c_1}{c_2} "
                f"reaches {ratio:.3e} of sqrt(g_{c_1}{c_1} "
                f"g_{c_2}{c_2}) on this grid: the off-diagonal "
                "rotation terms have no energy-conserving staggered "
                "form (the exact energy's cross term g_12 u^1 u^2 "
                "pairs values living at different nodes), so the "
                "module refuses rather than leak energy; use an "
                "orthogonal chart (lat-lon, torus, conformal maps)")

    # ================================================================
    #  The rotation term (linear)
    # ================================================================
    @term(advances=("u", "v"), linear=True, name="coriolis")
    def coriolis(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""``du = G vbar / W_1``; ``dv = -(G u)bar / W_2``.

        The class docstring's energy-conserving flux form: the flux
        weight :math:`G = f\,g\,w` is sampled once at the ``u``
        faces and averaged back to the ``v`` faces inside the flux
        (the shared modules' thickness-weighted pairing, with the
        metric folded into the weights) — exactly M-skew under
        :math:`\mathrm{diag}(W_1, W_2)`.

        The expression itself lives in `chart_rotation` (one owner:
        the shallow-water energy-correction module subtracts exactly
        this).
        """
        return chart_rotation(state, coords=self._coords,
                              metric_weight=self._metric_weight)
