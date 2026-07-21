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
import numbers
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
from fridom.model.params import (
    CORIOLIS_BETA,
    CORIOLIS_F0,
    CORIOLIS_METRIC_RATIO,
    CORIOLIS_ROSSBY,
    SCALING_NONLINEARITY,
)
from fridom.model.scheduled_field import ProfileFunction, profile_coords
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import term
from fridom.model.time_dependent import TimeDependent, resolve_at
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.space_patterns import Profile

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField

_U_HINT = ("velocities are declared by a dynamical-core module, "
           "e.g. nh.Core or sw.Core")


def _safe_metric_divide(
    num: ScalarField, den: ScalarField,
) -> ScalarField:
    r"""VJP-sealed metric divide ``num / den`` (masked ``0/0`` padding).

    Description
    -----------
    The energy-metric weight (``linear_rotation``'s ``w.to(v)``) and the
    chart metric weights (``chart_rotation``'s :math:`\sqrt g\,g_{ii}`)
    are an exact zero in the never-valid storage padding (and, for the
    velocity weight, the immersed dry cells): the outermost ring is
    stripped before any output, so the forward ``0/0`` there is
    harmless. Reverse-mode autodiff is not — the quotient VJP
    (:math:`-\mathrm{num}/\mathrm{den}^2` with ``den == 0``) turns the
    zero cotangent of a sealed cell into ``0 * inf = NaN`` and poisons
    every gradient with a data path through the rotation, the same
    masked singularity the Sadourny PV divide cures. Replacing the
    exact-zero denominators by 1 keeps the result finite there; valid
    cells (``den != 0``) divide by the true weight and are bitwise
    unchanged, forward and reverse (the ``bad`` mask covers only the
    padding). The same seal as ``advection._safe_ratio`` and the mapped
    pressure operator's ``_divide_by_jacobian`` (AGENTS.md diff policy).

    Parameters
    ----------
    num : ScalarField
        The flux numerator.
    den : ScalarField
        The energy / chart metric weight denominator.

    Returns
    -------
    ScalarField
        The ratio on the divide's structure, finite (0) in the
        never-valid padding.
    """
    if (getattr(num, "storage", None) is None
            or getattr(den, "storage", None) is None):
        # halo-trace stand-in (no data): the plain quotient flows the
        # space and ghost demand; the VJP seal is a runtime concern,
        # absent here (the metric-blind f-plane/beta-plane rotation is
        # halo-traced, so this branch fires under the halo accounting).
        return num / den
    bad = den.storage == 0.0
    safe = jnp.where(bad, 1.0, den.storage)
    ratio = jnp.where(bad, 0.0, num.storage / safe)
    # the field divide fixes the result's structure (space, merged halo
    # validity); its raw quotient data is discarded for the guarded
    # ratio, so the singular divide-VJP is never built.
    return (num / den).with_storage(ratio)


def linear_rotation(
    state: object,
    *,
    metric_weight: str | None = None,
    f_override: object = None,
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

    Returns
    -------
    dict
        The ``u`` / ``v`` increments.
    """
    u, v = state["u"], state["v"]
    # a scalar override is already constant in space (the f-plane), so
    # the field lift ``.to(u)`` is a no-op broadcast and is skipped;
    # otherwise the carried ``f_coriolis`` field is lifted onto the
    # ``u`` faces (a beta-plane ``FieldBlend`` rewrites that carried
    # field to the stage-time blend in its SELF_UPDATE stage — TDF-D11)
    f_u = (f_override if f_override is not None
           else state["f_coriolis"].to(u))
    if metric_weight is None:
        return {
            "u": f_u * v.to(u),
            "v": -((f_u * u).to(v)),
        }
    w = state[metric_weight]
    flux = (w.to(u) * f_u * u).to(v)
    return {
        "u": f_u * v.to(u),
        "v": -_safe_metric_divide(flux, w.to(v)),
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

    The state components ``u`` / ``v`` are the **physical** (m/s)
    velocities on every grid (``physical_state_components.md`` invariant
    (a)): the rotation converts them to the contravariant coordinate
    velocities :math:`u^i = U_i/\sqrt{g_{ii}}` at entry (VJP-sealed
    divide) and rescales the tendencies :math:`\mathrm{d}U_i =
    \sqrt{g_{ii}}\,\mathrm{d}u^i` at exit, returning **physical**
    increments; the M-skew flux form between the seams is unchanged. The
    metric factors are derived per call on each component's own bare
    staggered space, never cached.

    Parameters
    ----------
    state : VectorField
        The model state; reads the physical ``u``, ``v``,
        ``f_coriolis`` and (when named) the metric-weight field.
    coords : tuple[str, str]
        The chart coordinate names, in the grid's factor order.
    metric_weight : str | None, optional
        Name of the velocity energy-metric weight field ``w``
        (default: None).

    Returns
    -------
    dict
        The ``u`` / ``v`` increments (physical components).
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
    # entry seam: physical U -> contravariant u^i (D3); the M-skew
    # flux form below is unchanged, exit-rescaled to physical
    root_u = g_uu ** 0.5
    root_v = g_vv ** 0.5
    u = _safe_metric_divide(u, root_u)
    v = _safe_metric_divide(v, root_v)
    f_u = f.to(u)
    flux_weight = f_u * (sqg_u * sqg_u)          # G = f g (w below)
    w_1 = sqg_u * g_uu
    w_2 = sqg_v * g_vv
    if metric_weight is not None:
        w = state[metric_weight]
        flux_weight = flux_weight * w.to(u)
        w_1 = w_1 * w.to(u)
        w_2 = w_2 * w.to(v)
    # exit seam: rescale the contravariant tendencies to physical
    return {
        "u": root_u * _safe_metric_divide(flux_weight * v.to(u), w_1),
        "v": -(root_v * _safe_metric_divide(
            (flux_weight * u).to(v), w_2)),
    }


@term(advances=("u", "v"), linear=True, name="coriolis",
      linear_params=(CORIOLIS_F0, CORIOLIS_BETA, CORIOLIS_ROSSBY,
                     CORIOLIS_METRIC_RATIO, SCALING_NONLINEARITY),
      linear_fields=("f_coriolis",))
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
    beta-plane ``FieldBlend`` path, R2) or a ``ProfileFunction`` law
    (R3), the module rewrites the carried ``f_coriolis`` field to the
    stage-time value in a SELF_UPDATE stage (TDF-D11), so the term reads
    it plainly here — no term-side blend seam. A fully static module
    (plain-float ``f0`` and ``beta``) leaves ``f_override`` ``None`` and
    the assembly-frozen field path runs bit-identically.
    """
    stage_f = getattr(self, "_stage_scalar_f", None)
    f_override = stage_f(ctx) if stage_f is not None else None
    body = linear_rotation(
        state, metric_weight=self._metric_weight,
        f_override=f_override)
    if not getattr(self, "_nondim", False):
        return body
    # nondimensional variant: the carried f_coriolis is the SHAPE
    # (1, or 1 + metric_ratio * y) and the whole rotation is scaled
    # by the live ratio epsilon / Ro (stage-time ctx.params reads;
    # under the matching Rotational scaling the alias row makes the
    # ratio an exact 1.0)
    scale = (ctx.params[SCALING_NONLINEARITY]
             / ctx.params[CORIOLIS_ROSSBY])
    return {name: scale * value for name, value in body.items()}


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
#: read at stage time (AR-D2). A SELF_UPDATE stage (TDF-D11) rewrites the
#: carried ``f_coriolis`` field to this blend each substage, so every
#: reader (the rotation term, I/O, restart) sees the stage-time value.
#: The two-endpoint paper form ``f0 + rho(t/tau)*beta*y`` is the special
#: case ``f0`` static, ``beta`` a ``Ramp`` (its ``0 -> beta`` ramp is
#: ``rho(t/tau)*beta``).
_BETA_BLEND = FieldBlend((
    BlendIngredient("f_coriolis_const", weight="f0",
                    build=_f_const_ingredient),
    BlendIngredient("f_coriolis_grad", weight="beta",
                    build=_f_grad_ingredient),
))


def _check_rotation_kwargs(
    name: str, dim_value: object, rossby_number: object,
) -> None:
    """Validate the dual kwarg sets of the Coriolis family.

    Exactly one of the dimensional leaf (``f0``) and the
    nondimensional ``rossby_number`` must be given, and the Rossby
    number must be nonzero (the live rotation ratio divides by it —
    cheaper refused at construction than guarded per step).
    """
    if (dim_value is None) == (rossby_number is None):
        raise TypeError(
            f"{name} takes exactly one kwarg set: DIMENSIONAL "
            "f0= (physical rotation, zero scaling ops in the "
            "trace) XOR NONDIMENSIONAL rossby_number= (the "
            "rotation scaled by the live epsilon/Ro ratio, under a "
            "nondimensional fr.scaling policy); got "
            f"f0={dim_value!r}, rossby_number={rossby_number!r}")
    if (isinstance(rossby_number, numbers.Number)
            and float(rossby_number) == 0.0):
        raise TypeError(
            f"{name} rossby_number=0 is refused: the rotation "
            "carries the live ratio epsilon/Ro, which divides by "
            "it; pass a nonzero Rossby number")


def _reject_chart_grid(module: Module, table: object) -> None:
    """Refuse a metric-blind rotation on a chart-coupled grid.

    The f-plane and beta-plane terms rotate the velocity components
    as if they were Cartesian (no ``sqrt(g)``, no ``g_ij``): on a
    chart grid the rotation must carry the metric to stay
    energy-conserving, so the metric-blind term would be silently wrong
    physics (and would do work against the metric energy). Better a
    taught error than a plausible-looking wrong answer.
    """
    chart = table.grid.chart_coords
    if chart is None:
        return
    raise ValueError(
        f"{type(module).__name__} is metric-blind (it rotates "
        "Cartesian velocity components), but this grid carries an "
        f"embedding chart on {chart}, whose rotation needs the "
        "metric-aware chart form: use "
        "fr.modules.RotationCoriolis(omega=(0.0, 0.0, Omega), "
        f"coords={chart!r}) — it derives f = 2 Omega . n_hat from "
        "the chart itself, and the lat-lon sphere with a polar Omega "
        "gives f = 2 Omega sin(lat) — or omit coriolis= entirely to "
        "run without rotation")


@partial(jaxify, dynamic=("f0", "rossby_number"))
class FPlaneCoriolis(Module):

    r"""
    Constant-rotation Coriolis on the f-plane (dual scaling variants).

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on ``fr.Profile()``
    (one degree of freedom) and carries the linear rotation term.
    The two mutually-exclusive kwarg sets fix the **variant** at
    construction (``fr.scaling``):

    - **dimensional** (``f0=``): :math:`f \equiv f_0` everywhere,
      the rotation term verbatim (zero scaling ops); provides the
      constant ``coriolis.f0`` (provides-implies-constancy);
    - **nondimensional** (``rossby_number=``): the field is the
      f-shape ``1`` and the rotation is scaled by the live ratio
      :math:`\varepsilon/\mathrm{Ro}`; provides
      ``coriolis.rossby``. As the ``rotation`` mechanism owner, the
      assembly aliases ``scaling.nonlinearity`` onto this leaf under
      ``fr.scaling.Rotational()``.

    Parameters
    ----------
    f0 : float | fr.Ramp | None, optional
        The constant Coriolis parameter :math:`f_0` [1/s]
        (dimensional variant) (default: None).
    rossby_number : float | fr.Ramp | None, optional
        The Rossby number :math:`\mathrm{Ro}` (nondimensional
        variant); must be nonzero (the live ratio divides by it)
        (default: None).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``); switches
        the rotation to the thickness-weighted flux form, the
        M-skew pairing under ``diag(w, w, ...)`` (default: None).
    """

    #: fr.scaling traits: this family owns the rotation mechanism
    scaling_mechanism = "rotation"
    nonlinearity_attr = "rossby_number"

    def __init__(
        self, f0: float | None = None, *,
        rossby_number: float | None = None,
        metric_weight: str | None = None,
    ) -> None:
        """Store the variant's leaf (exactly one kwarg set).

        Raises
        ------
        TypeError
            If both or neither of ``f0``/``rossby_number`` are
            given, or ``rossby_number`` is exactly zero.
        """
        _check_rotation_kwargs(type(self).__name__, f0,
                               rossby_number)
        self.f0 = None if f0 is None else leaf(f0)
        self.rossby_number = (None if rossby_number is None
                              else leaf(rossby_number))
        self._nondim: bool = rossby_number is not None
        self._metric_weight = metric_weight

    @property
    def scaling_variant(self) -> str:
        """The constructor-fixed variant (``fr.scaling`` seam)."""
        return "nondimensional" if self._nondim else "dimensional"

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """``coriolis.f0`` (dim) / ``coriolis.rossby`` (nondim)."""
        if self._nondim:
            return (ParameterDeclaration(
                CORIOLIS_ROSSBY, attr="rossby_number", units="1",
                doc="Rossby number (the rotation mechanism)"),)
        return (ParameterDeclaration(
            CORIOLIS_F0, attr="f0", units="1/s",
            doc="constant Coriolis parameter"),)

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
        value = (1.0 if self._nondim
                 else resolve_at(self.f0, 0.0))
        return grid.create_field(
            space, data=jnp.full(space.shape, value),
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
        ``scaling.nonlinearity`` ramp rides), so the value is correct in the
        assembly dry run, ``model.tendency`` and every stepper stage
        alike. Returns ``None`` for a plain-float ``f0`` (the field
        path stays bit-identical); the static branch never touches
        ``ctx`` (host-side dispatch on the leaf type), so the term is
        still callable with ``ctx=None``.
        """
        if isinstance(self.f0, TimeDependent):
            return ctx.params[CORIOLIS_F0]
        return None  # nondim shape field is static (f0 is None)

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


def _check_profile_law(
    f: object, f0: object, beta: object,
) -> ProfileFunction | None:
    """Validate a law-valued ``f`` (the general ``f(y,t)`` path, TDF-D7).

    ``f`` is a ``ProfileFunction`` giving the complete Coriolis field
    ``f(y,t)`` — it supersedes ``f0``/``beta`` (which stay inert
    leaves), so a *time-dependent* ``f0``/``beta`` alongside it is an
    ambiguous double source and is rejected. ``None`` keeps the affine
    ``f0 + beta*y`` path unchanged.
    """
    if f is None:
        return None
    if not isinstance(f, ProfileFunction):
        raise TypeError(
            f"f={f!r} must be a fr.model.ProfileFunction giving the full "
            "f(y,t) law; a constant/affine Coriolis parameter is the "
            "f0/beta path")
    if isinstance(f0, TimeDependent) or isinstance(beta, TimeDependent):
        raise TypeError(
            "a ProfileFunction f is the complete f(y,t) law and "
            "supersedes f0/beta; it cannot be combined with a "
            "time-dependent (Ramp) f0/beta — pass one or the other")
    return f


@partial(jaxify, dynamic=("f0", "beta", "rossby_number",
                          "metric_ratio", "_f_law"))
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
    ``f_coriolis_grad``, and ``f_coriolis`` is marked ``time_dependent``:
    a SELF_UPDATE stage rewrites it to the stage-time blend every
    substage (TDF-D11), so the rotation term, I/O and any cross-module
    reader all see the fresh value (no frozen ``t = 0`` snapshot).

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
    f : fr.model.ProfileFunction | None, optional
        A complete non-affine law :math:`f(y,t)` (TDF-D7); supersedes
        ``f0``/``beta`` and drives a ``time_dependent`` ``f_coriolis``
        field rewritten each substage (default: None).
    """

    #: fr.scaling traits: this family owns the rotation mechanism
    scaling_mechanism = "rotation"
    nonlinearity_attr = "rossby_number"

    def __init__(
        self, f0: float | None = None, beta: float | None = None,
        *, rossby_number: float | None = None,
        metric_ratio: float | None = None,
        meridional: str = "y", metric_weight: str | None = None,
        f: ProfileFunction | None = None,
    ) -> None:
        r"""Store the leaves and the meridional coordinate name.

        A time-dependent ``f0`` and/or ``beta`` (an ``fr.Ramp``) is
        supported through the beta-plane ``FieldBlend`` (AR-D2 / R2):
        the Coriolis parameter is the spatially varying field
        :math:`f(y,t) = f_0(t) + \beta(t)\,y`, blended at stage time
        from the assembly-materialized unit and meridional-coordinate
        profiles (see :data:`_BETA_BLEND`) and written back to the
        carried ``f_coriolis`` field each substage by a SELF_UPDATE
        stage (TDF-D11). The static (plain-float) path is untouched.

        A ``ProfileFunction`` ``f`` gives the complete **non-affine**
        law :math:`f(y,t)` (TDF-D7): the ``f_coriolis`` field is marked
        ``time_dependent`` and rewritten every substage by a SELF_UPDATE
        stage sampling the law at the stage clock; ``f0``/``beta`` become
        inert. A frozen-``L`` (ETDRK4) stepper then refuses the model
        automatically (the marker feeds the frozen-``L`` guard).

        The NONDIMENSIONAL kwarg set (``rossby_number=`` + optional
        ``metric_ratio=``) replaces ``f0``/``beta``: the carried
        field is the f-shape ``1 + metric_ratio * y`` and the
        rotation is scaled by the live epsilon/Ro ratio. A ramped
        ``metric_ratio`` is refused (the shape field carries no
        blend for it — a recorded follow-up); ramp ``rossby_number``
        instead (a stage-time ``ctx.params`` read, no field
        rewrite needed).

        Raises
        ------
        TypeError
            On a mixed kwarg set, a zero ``rossby_number``, a
            time-dependent ``metric_ratio``, or an ``f`` law next to
            the nondimensional set.
        """
        _check_rotation_kwargs(type(self).__name__, f0,
                               rossby_number)
        if rossby_number is not None:
            if beta is not None:
                raise TypeError(
                    "BetaPlaneCoriolis beta= belongs to the "
                    "DIMENSIONAL kwarg set (f0= + beta=); the "
                    "nondimensional set spells the meridional "
                    "variation as metric_ratio= (f-shape "
                    "1 + metric_ratio * y)")
            if f is not None:
                raise TypeError(
                    "BetaPlaneCoriolis f= (the full f(y,t) law) is "
                    "the DIMENSIONAL path; the nondimensional set "
                    "takes rossby_number= + metric_ratio=")
            if isinstance(metric_ratio, TimeDependent):
                raise TypeError(
                    "BetaPlaneCoriolis metric_ratio= is "
                    "time-dependent, but the nondimensional f-shape "
                    "field carries no blend for it (a recorded "
                    "follow-up); ramp rossby_number= instead — the "
                    "rotation reads it from ctx.params at stage "
                    "time")
            metric_ratio = 0.0 if metric_ratio is None else metric_ratio
        else:
            if metric_ratio is not None:
                raise TypeError(
                    "BetaPlaneCoriolis metric_ratio= belongs to the "
                    "NONDIMENSIONAL kwarg set (rossby_number= + "
                    "metric_ratio=); the dimensional set spells the "
                    "meridional variation as beta=")
            beta = 0.0 if beta is None else beta
        self.f0 = None if f0 is None else leaf(f0)
        self.beta = None if beta is None else leaf(beta)
        self.rossby_number = (None if rossby_number is None
                              else leaf(rossby_number))
        self.metric_ratio = (None if metric_ratio is None
                             else leaf(metric_ratio))
        self._nondim: bool = rossby_number is not None
        self._meridional = meridional
        self._metric_weight = metric_weight
        self._f_law = (None if self._nondim else
                       _check_profile_law(f, self.f0, self.beta))
        #: grid coordinate names for the profile-path halo (set at bind)
        self._halo_coords: tuple[str, ...] = ()

    @property
    def scaling_variant(self) -> str:
        """The constructor-fixed variant (``fr.scaling`` seam)."""
        return "nondimensional" if self._nondim else "dimensional"

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """``coriolis.beta`` (dim) / rossby + metric_ratio (nondim).

        The dimensional variant deliberately does **not** provide
        ``coriolis.f0`` (its Coriolis parameter is the ``f(y)``
        field, not a constant — 02_rules).
        """
        if self._nondim:
            return (
                ParameterDeclaration(
                    CORIOLIS_ROSSBY, attr="rossby_number",
                    units="1",
                    doc="Rossby number (the rotation mechanism)"),
                ParameterDeclaration(
                    CORIOLIS_METRIC_RATIO, attr="metric_ratio",
                    units="1",
                    doc="meridional f-shape gradient (nondim)"),
            )
        return (
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
    def _profile_active(self) -> bool:
        """Whether a ``ProfileFunction`` drives ``f(y,t)`` (TDF-D7).

        A host-side (structural) predicate on the ``f`` law type; the
        general non-affine path is mutually exclusive with the affine
        ``FieldBlend`` (``_check_profile_law`` rejects a Ramped
        ``f0``/``beta`` alongside a law).
        """
        return self._f_law is not None

    @property
    def extra_halo(self) -> HaloSpec | None:
        """The profile-path halo substitute (V-N2); ``None`` otherwise.

        The SELF_UPDATE rewrites ``f_coriolis`` from raw sampled data
        (``with_data``, halo-trace exempt), so on the law path the module
        declares its rotation-term reach itself: one ghost cell per grid
        axis covers the term's staggered ``.to`` averages (reach 1). The
        static / affine-blend paths keep ``None`` and stay halo-traced
        bit-identically.
        """
        if not self._profile_active:
            return None
        return HaloSpec(dict.fromkeys(self._halo_coords, 1))

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The ``f(y)`` field, plus the blend ingredients when ramped.

        The static (plain-float) path declares the single ``f_coriolis``
        profile exactly as before (``time_dependent=False``, so the
        assembly fingerprint is untouched — the marker is repr-
        participating only when True). A ramped ``f0``/``beta`` marks
        ``f_coriolis`` ``time_dependent`` and additionally declares the
        two ``FieldBlend`` ingredient profiles (``f_coriolis_const``,
        ``f_coriolis_grad``); a SELF_UPDATE stage rewrites ``f_coriolis``
        to the stage-time blend each substage (TDF-D11), so every reader
        sees the fresh value. A ``ProfileFunction`` ``f`` (TDF-D7) marks
        the single ``f_coriolis`` field ``time_dependent`` (materialized
        at ``t = 0``), rewritten each substage by its own SELF_UPDATE
        stage.
        """
        if self._profile_active:
            return (FieldDeclaration(
                "f_coriolis", space=Profile(self._meridional),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._f_profile_default,
                long_name="Coriolis parameter", units="1/s",
                time_dependent=True),)
        f_coriolis = FieldDeclaration(
            "f_coriolis", space=Profile(self._meridional),
            lifecycle=Lifecycle.AUXILIARY, default=self._f_default,
            long_name="Coriolis parameter", units="1/s",
            time_dependent=self._blend_active)
        if not self._blend_active:
            return (f_coriolis,)
        return (f_coriolis, *_BETA_BLEND.field_declarations(
            space=Profile(self._meridional),
            long_name="Coriolis parameter blend ingredient",
            units="1/s"))

    def _f_profile_default(
        self, grid: object, space: object,
    ) -> ScalarField:
        r"""Owner-method default: sample the ``f(y,t)`` law at ``t = 0``.

        Mirrors ``_f_default``'s ``resolve_at(..., 0.0)`` spelling for
        the law path: the AUXILIARY field is materialized as the ``t = 0``
        snapshot so it keeps a valid static treedef; the SELF_UPDATE
        stage then rewrites it with the stage-time value each substage,
        so this frozen value is never read at run time. No pre-syncing
        (GAP-B).
        """
        coords = profile_coords(grid, space, (self._meridional,))
        data = self._f_law.sample(coords, 0.0, space.shape)
        return grid.create_field(space, data=data, name="f_coriolis")

    # ================================================================
    #  The SELF_UPDATE stage (law or blend path, S1 per substage)
    # ================================================================
    @property
    def stages(self) -> tuple[Stage, ...]:
        """The per-substage ``f_coriolis`` rewrite (law or blend path).

        The law path (``ProfileFunction`` ``f``, TDF-D7) recomputes the
        full ``f(y,t)`` law from raw sampled data; the blend path (a
        ramped ``f0``/``beta``, AR-D2 / TDF-D11) rewrites the carried
        field to the affine ``FieldBlend`` combination via the reusable
        ``FieldBlend.stage`` helper. The two are mutually exclusive
        (``_check_profile_law`` rejects a ramped ``f0``/``beta`` beside a
        law), and a fully static module emits no stage.
        """
        if self._profile_active:
            return (Stage(
                kind=StageKind.SELF_UPDATE, fn="_update_f_coriolis",
                name="coriolis_f", reads=("f_coriolis",),
                writes=("f_coriolis",)),)
        if self._blend_active:
            return (_BETA_BLEND.stage(
                "_update_f_coriolis_blend", target="f_coriolis",
                name="coriolis_f_blend"),)
        return ()

    def _update_f_coriolis(self, state: object, ctx: object) -> dict:
        """Re-evaluate the ``f(y,t)`` law at the substage clock (TDF-D7).

        SELF_UPDATE runs first in every substage (S1), so the rotation
        term (and every ``f_coriolis`` consumer) reads the stage-time
        field, consistent with ``eval_params``.
        """
        time = getattr(ctx.clock, "time", ctx.clock)
        field = state["f_coriolis"]
        space = field.function_space
        coords = profile_coords(field.grid, space, (self._meridional,))
        value = self._f_law.sample(coords, time, space.shape)
        return {"f_coriolis": field.with_data(value)}

    def _update_f_coriolis_blend(
        self, state: object, ctx: object,
    ) -> dict:
        r"""Rewrite ``f_coriolis`` to the stage-time ``FieldBlend`` (D11).

        The blend-path counterpart of ``_update_f_coriolis``: it
        delegates to the reusable ``FieldBlend.rewrite`` core, which
        evaluates :math:`f_0(t) + \beta(t)\,y` from the ingredient
        profiles at the substage clock and returns it as a full-field
        write. SELF_UPDATE runs first in every substage (S1), so the
        rotation term reads the fresh blend.
        """
        return _BETA_BLEND.rewrite(self, state, ctx, target="f_coriolis")

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: materialize ``f0 + beta*y`` at ``t=0``.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``.
        A time-dependent ``f0``/``beta`` (an ``fr.Ramp``) is resolved at
        ``t = 0`` (``resolve_at``) so the AUXILIARY field keeps a valid
        static treedef; the SELF_UPDATE stage then rewrites it to the
        stage-time blend each substage (TDF-D11), so this frozen value is
        only a placeholder on the ramped path. A plain-float
        ``f0``/``beta`` is ``resolve_at``-identity, so this line is
        bit-identical to the static case. No pre-syncing (GAP-B) — see
        ``FPlaneCoriolis._f_default``.
        """
        if self._nondim:
            f0 = 1.0
            beta = resolve_at(self.metric_ratio, 0.0)
        else:
            f0 = resolve_at(self.f0, 0.0)
            beta = resolve_at(self.beta, 0.0)
        mer = self._meridional

        def init(**coords: object) -> object:
            return f0 + beta * coords[mer]

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="f_coriolis")

    def bind(self, table) -> None:  # noqa: ANN001
        """Reject chart-coupled grids; record the law-path halo axes.

        Raises
        ------
        ValueError
            If the grid carries an embedding chart.
        """
        _reject_chart_grid(self, table)
        if self._profile_active:
            self._halo_coords = tuple(table.grid.names)

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
    tangent plane. The derivation below is written in **contravariant**
    coordinate velocities :math:`u^i = \dot u^i`; the stored state
    components are the **physical** velocities
    :math:`U_i = \sqrt{g_{ii}}\,u^i`
    (``physical_state_components.md`` invariant (a)), which
    :func:`chart_rotation` converts to contravariant at entry and back
    at exit. Using
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
    @term(advances=("u", "v"), linear=True, name="coriolis",
          linear_fields=("f_coriolis",))
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
