r"""
Harmonic and biharmonic diffusion closures (mixing and friction).

Description
-----------
The framework-level generic diffusion closure family (the port of
the old ``fr.modules.closures.HarmonicDiffusion`` /
``BiharmonicDiffusion`` and their per-model mixing/friction wrappers).
The harmonic operator on a target :math:`q` is

.. math::
    \mathcal{H}(q) = \nabla \cdot (\mathbf{A} \cdot \nabla q),
    \qquad \mathbf{A} = \mathrm{diag}(\kappa_1, \dots, \kappa_n),

spelled as pure field arithmetic — per axis, a staggered first
difference, the coefficient inside the flux, and the difference back
onto :math:`q`'s space — so the stencils are halo-traced normally.
Following Griffies & Hallberg (2000), the biharmonic operator
iterates the harmonic one with square-rooted coefficients,

.. math::
    \mathcal{B}(q) = -\mathcal{H}_{\sqrt{\kappa}}(
        \mathcal{H}_{\sqrt{\kappa}}(q)),

so the user-facing :math:`\kappa_i \ge 0` is the true biharmonic
coefficient (units :math:`\mathrm{m^4/s}`).

Targets are role-driven (D1.4, `ClosureBase`): the mixing classes
default to ``fr.roles.TRACER``, the friction classes to the
``Velocity`` family; ``fields=`` / ``exclude=`` override by name. The
old ``ENABLE_MIXING`` / ``ENABLE_FRICTION`` flags are gone.
Coefficients are **provided parameters** (dynamic leaves) published
under the canonical names below, so ``model.update_parameters``
sweeps them without a recompile; each accepts a scalar (or
``fr.Ramp``) or a name-keyed per-field mapping (plain numbers),
validated at bind. Anisotropy is horizontal/vertical: the optional
``*_v`` coefficient acts along the ``vertical`` coordinate (default
``"z"``), the main coefficient along every other axis (and along all
axes when ``*_v`` is omitted) — the old ``(kh, kh, kv)`` /
``(ah, av)`` spellings.

**Walled grids** (bounded mesh factors) are supported per axis. The
two-pass flux chain cannot close on a bounded BC-free face (R1: the
second difference needs the wall-face flux, which a BC-free ``Inner``
does not define), so along a bounded axis the interior flux is
retagged onto its Dirichlet-tagged ``Inner`` sibling — a true
wall-value claim, the wall flux *is* zero — and the closing
difference telescopes with that structural exact-zero wall flux (the
advection flux-space BC-sibling precedent). This is the **no-flux**
tracer wall and the **free-slip** velocity wall (zero tangential wall
stress). The wall-normal velocity component (staggered along the
walled axis, space ``Inner[Dirichlet]``) closes on its own tag with
no slip choice. The **no-slip** velocity wall (``slip="no"`` on the
friction closures) adds the wall-adjacent-cell correction
:math:`-2\,\nu\,u_1/\Delta n^2` — the factor-of-2 ghost against the
zero wall value across the half cell — on top of the free-slip chain.
Periodic axes take the exact periodic path (bit-identical to a
purely periodic grid). The biharmonic closures apply the same wall
treatment on both Laplacian passes (Griffies & Hallberg). The walled
support is the **nodal** family (Center cells, Inner velocity faces);
a finite-volume ``CellAvg`` target on a walled axis is rejected at
bind (FV walled closures are future work).

**Mapped / terrain-following grids** (stretched ``MappedIntervalMesh``
factors, or a ``Grid(..., mapping=CoordinateMapping(maps=...))`` chart)
bind and run through the same per-axis chain: the operator is
**along-coordinate** (the ROMS-default ``MIX_S_UV`` convention). Each
``q.diff(axis)`` is the *computational* (along-coordinate) derivative,
which divides by the codomain's own measure (the two-point node
spacing, order 2), so:

- On a **stretched** bounded column the measure *is* the physical cell
  spacing, so the chain is the physical :math:`\partial_z(k\,\partial_z
  q)` and is **metrically exact at order 2** (the same-row-Jacobian
  argument, ``stretched_terrain_combined.md`` §5); the measure-weighted
  telescoping keeps the no-flux tracer integral machine-zero.
- On a **terrain-following** chart the operator acts on
  *constant-coordinate* (constant-:math:`\sigma`) surfaces: the chart
  factor :math:`H(x)` lives in ``grid.metric`` and **never enters**
  ``diff``/``measure``, so the per-axis legs carry no cross terms and
  no :math:`H(x)` coupling — "horizontal" mixing **tilts with the
  terrain**. This is accepted practice for **viscosity** (ROMS
  default), but a physics error for **tracer** mixing over steep
  slopes (spurious diapycnal mixing); the geopotential/rotated
  operator that fixes tracer orientation is future work (record §3.6
  options A/C). The ``vertical=`` split is by **axis name**, so on a
  terrain grid ``nu_v``/``kappa_v`` (with e.g. ``vertical="sigma"``)
  acts along the **column coordinate**, not the physical vertical.

Forward is exact as above, and reverse-mode ``jax.grad`` is finite on
every supported grid kind — the stretched-column boundary-face
measure divide is VJP-sealed in the spatial layer
(``divide_by_codomain_measure``); the mirrored test shards carry the
autodiff regressions.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from fridom.framework.utils import jaxify, modify_array
from fridom.model.closures.base import ClosureBase
from fridom.model.errors import AssemblyError
from fridom.model.parameters import (
    ParameterDeclaration,
    leaf,
)
from fridom.model.params import ParamName
from fridom.model.roles import TRACER, Velocity
from fridom.model.terms import TendencyTerm, Treatment
from fridom.model.time_dependent import TimeDependent
from fridom.spatial.bc import BC
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from fridom.model.context import StepContext
    from fridom.model.field_table import FieldTable
    from fridom.model.roles import Role
    from fridom.spatial.fields.scalar_field import ScalarField


# ================================================================
#  Canonical parameter names (one provider per name)
# ================================================================
MIXING_KAPPA: ParamName = ParamName(
    "mixing.kappa", units="m^2/s",
    hint="provided by fr.closures.HarmonicDiffusion(kappa=...)")

MIXING_KAPPA_V: ParamName = ParamName(
    "mixing.kappa_v", units="m^2/s",
    hint="provided by fr.closures.HarmonicDiffusion(kappa_v=...)")

MIXING_KAPPA4: ParamName = ParamName(
    "mixing.kappa4", units="m^4/s",
    hint="provided by fr.closures.BiharmonicDiffusion(kappa=...)")

MIXING_KAPPA4_V: ParamName = ParamName(
    "mixing.kappa4_v", units="m^4/s",
    hint="provided by fr.closures.BiharmonicDiffusion(kappa_v=...)")

FRICTION_NU: ParamName = ParamName(
    "friction.nu", units="m^2/s",
    hint="provided by fr.closures.HarmonicFriction(nu=...)")

FRICTION_NU_V: ParamName = ParamName(
    "friction.nu_v", units="m^2/s",
    hint="provided by fr.closures.HarmonicFriction(nu_v=...)")

FRICTION_NU4: ParamName = ParamName(
    "friction.nu4", units="m^4/s",
    hint="provided by fr.closures.BiharmonicFriction(nu=...)")

FRICTION_NU4_V: ParamName = ParamName(
    "friction.nu4_v", units="m^4/s",
    hint="provided by fr.closures.BiharmonicFriction(nu_v=...)")


# ================================================================
#  Coefficient coercion (construction-time, host-side)
# ================================================================
def _coerce_scalar(
    value: object, arg: str, owner: str, *, nonnegative: bool,
) -> object:
    """Coerce one scalar coefficient; validate biharmonic signs."""
    if isinstance(value, TimeDependent):
        return value
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(
            f"{owner}: {arg}= takes a number, an fr.Ramp, or a "
            f"name-keyed per-field mapping; got {value!r}") from None
    if nonnegative and numeric < 0.0:
        raise ValueError(
            f"{owner}: {arg}= must be non-negative, got {numeric!r}"
            " (the biharmonic operator is -H(H(q)) with "
            "square-rooted coefficients; a negative coefficient "
            "has no square root)")
    return leaf(value)


def _coerce_coefficient(
    value: object, arg: str, owner: str, *, nonnegative: bool,
) -> object:
    """Coerce a scalar-or-mapping coefficient to dynamic leaves."""
    if not isinstance(value, Mapping):
        return _coerce_scalar(value, arg, owner,
                              nonnegative=nonnegative)
    if not value:
        raise ValueError(
            f"{owner}: {arg}= mapping is empty; give every target "
            "a coefficient, or pass a scalar")
    coerced: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"{owner}: {arg}= mapping keys are field names, "
                f"got {key!r}")
        if isinstance(entry, TimeDependent):
            raise TypeError(
                f"{owner}: per-field {arg}= mappings take plain "
                "numbers; a ramped coefficient needs the scalar "
                "spelling (one closure per ramped target)")
        coerced[key] = _coerce_scalar(
            entry, f"{arg}[{key!r}]", owner, nonnegative=nonnegative)
    return coerced


# ================================================================
#  Slip validation (construction-time, host-side)
# ================================================================
#: the wall-stress vocabulary of the friction closures
FREE_SLIP: str = "free"
NO_SLIP: str = "no"
_SLIP_VALUES: tuple[str, str] = (FREE_SLIP, NO_SLIP)


def _check_slip_value(value: object, arg: str, owner: str) -> str:
    """Validate one slip choice against the vocabulary."""
    if value not in _SLIP_VALUES:
        raise ValueError(
            f"{owner}: {arg}= must be {FREE_SLIP!r} (zero tangential "
            f"wall stress) or {NO_SLIP!r} (u = 0 at the wall via the "
            f"factor-of-2 ghost), got {value!r}")
    return value  # type: ignore[return-value]


def _coerce_slip(value: object, owner: str) -> str | dict[str, str]:
    """Coerce ``slip=`` to a scalar choice or a per-field mapping."""
    if not isinstance(value, Mapping):
        return _check_slip_value(value, "slip", owner)
    if not value:
        raise ValueError(
            f"{owner}: slip= mapping is empty; give every target a "
            "slip choice, or pass a scalar 'free'/'no'")
    coerced: dict[str, str] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"{owner}: slip= mapping keys are field names, got "
                f"{key!r}")
        coerced[key] = _check_slip_value(entry, f"slip[{key!r}]", owner)
    return coerced


# ================================================================
#  Per-axis wall treatment (bind-time classification)
# ================================================================
#: the three per-axis wall regimes of the flux chain
_PERIODIC: str = "periodic"
_WALL_NORMAL: str = "wall_normal"
_TANGENTIAL: str = "tangential"


def _wall_treatment(
    factor: object, owner: str, name: str, axis: str,
) -> str:
    """Classify one target factor along its coordinate axis.

    Periodic factors take the exact periodic path; a Dirichlet-tagged
    ``Inner`` face (the wall-normal velocity) closes on its own tag; a
    BC-free ``Center`` cell field (a tracer, a tangential velocity)
    takes the flux-retag wall closure. Any other placement/BC — a
    finite-volume ``CellAvg`` target, a fixed-value (Dirichlet) cell
    wall, a bare face — is out of scope and rejected loudly.
    """
    if getattr(factor.mesh, "periodic", False):
        return _PERIODIC
    node_set = getattr(factor, "node_set", None)
    has_dirichlet = BC.DIRICHLET in factor.bc.components
    if node_set is NodeSet.INNER and has_dirichlet:
        return _WALL_NORMAL
    if node_set is NodeSet.CENTER and factor.bc.is_free:
        return _TANGENTIAL
    raise NotImplementedError(
        f"{owner}: target {name!r} has an unsupported wall placement "
        f"along {axis!r} ({factor!r}); the walled closure is the "
        "nodal family — no-flux / free-slip / no-slip cell fields "
        "(BC-free Center) and the wall-normal velocity face "
        "(Inner[Dirichlet]). A finite-volume (CellAvg) target, a "
        "fixed-value (Dirichlet) cell wall, or a bare face is out of "
        "scope (FV walled closures / the stage-2e inhomogeneous "
        "boundary-data path are future work)")


# ================================================================
#  The shared harmonic pass (pure field arithmetic)
# ================================================================
def _dirichlet_face(flux: ScalarField, axis: str) -> object:
    """Return the Dirichlet BC-sibling of the flux's ``Inner`` factor.

    The first difference of a BC-free ``Center`` field lands on the
    BC-free nodal ``Inner`` face, so the sibling is the same interior
    face with the Dirichlet tag (the wall flux is a structural zero).
    """
    factor = flux.function_space.bare.factor(axis)
    return factor.mesh.nodal(factor.node_set, bc=BC.DIRICHLET)


def _wall_correction(q: ScalarField, k: object, axis: str) -> ScalarField:
    r"""No-slip wall-adjacent tendency correction along ``axis``.

    Adds :math:`-2\,k\,q/\Delta n^2` at both wall-adjacent cells (index
    0 and -1 along ``axis`` — an index test, never a coordinate-value
    test): the factor-of-2 ghost (``q_ghost = -q_1`` across the wall,
    wall value 0) makes the wall-face flux :math:`2 k q_1/\Delta n`,
    whose flux-difference contribution in the wall cell is
    :math:`-2 k q_1/\Delta n^2`. Both lengths are the wall cell's own
    width, read from ``grid.measure`` (so stretched columns are
    correct); the row is static and homogeneous-linear in ``q``, so it
    is jit-pure and reverse-mode differentiable with no singularity.

    Built fresh from ``q``'s (negotiated) grid at evaluation, so the
    weight shares the step's storage frame. The halo tracer carries no
    grid, so the caller skips this pointwise, zero-reach correction
    during negotiation (see :func:`_harmonic`).

    The :math:`1/\Delta n^2` factor is folded into the true-shape
    weight array (cell widths are positive on the primal centres, so
    no division by zero) rather than divided in as a field: a
    field-level ``/ measure`` divides in the never-synced ghost slots
    too, where the zero-weight numerator makes the reverse pass a
    ``0/0`` NaN even though the forward value is masked.
    """
    grid = q.grid
    space = q.function_space
    bare = space.bare
    ax = bare.names.index(axis)
    ndim = len(bare.names)
    indicator = jnp.zeros(space.shape)
    for end in (0, -1):
        where = tuple(end if a == ax else slice(None)
                      for a in range(ndim))
        indicator = modify_array(indicator, where, 1.0)
    width = grid.measure(space, axis).data  # true-shape primal widths
    weight = grid.create_field(space, data=indicator / (width * width))
    return q * weight * (-2.0 * k)


def _harmonic(
    q: ScalarField,
    axes: tuple[tuple[str, object, str], ...],
    no_slip: frozenset[str],
) -> ScalarField:
    r"""One ``div(A grad q)`` pass with the per-axis wall treatment.

    ``axes`` is ``(axis, coefficient, treatment)`` per coordinate, in
    horizontal-then-vertical order (the float-summation order of the
    purely periodic path, preserved bit-for-bit). Periodic axes take
    the plain two-pass chain; bounded tangential/tracer axes retag the
    interior flux onto its Dirichlet sibling (structural zero wall
    flux) and, on a no-slip axis, add the wall-adjacent correction;
    bounded wall-normal axes close on the target's own tag and retag
    the result back onto it.
    """
    res = None
    for axis, k, treatment in axes:
        if treatment == _PERIODIC:
            contribution = (q.diff(axis) * k).diff(axis)
        elif treatment == _WALL_NORMAL:
            contribution = (q.diff(axis) * k).diff(axis).retag(q)
        else:  # tangential / tracer: retag the interior flux
            flux = q.diff(axis) * k
            flux = flux.retag(_dirichlet_face(flux, axis))
            contribution = flux.diff(axis)
            # the correction is pointwise (zero halo reach); skip it on
            # the grid-less halo tracer, materialize it on the real grid
            if axis in no_slip and hasattr(q.grid, "create_field"):
                contribution = contribution + _wall_correction(
                    q, k, axis)
        res = (contribution if res is None
               else res + contribution)
    return res


def _biharmonic_root(coeff: object) -> object:
    r"""Square-root split of a biharmonic coefficient, AD-safe at zero.

    The biharmonic operator applies ``sqrt(coeff)`` per Laplacian pass.
    Reverse-mode AD: the ``sqrt`` VJP ``0.5 * coeff**-0.5`` is ``inf`` at
    ``coeff = 0``, so a plain ``coeff ** 0.5`` returns ``NaN`` from
    ``jax.grad`` at ``nu4 = 0`` even though the forward value is finite.
    The double ``jnp.where`` guards both branches: the primal is bitwise
    identical for ``coeff > 0`` (the taken branch), and the measure-zero
    subgradient at ``coeff = 0`` is pinned to ``0``.

    Demoted Python scalars (halo trace) keep the plain ``** 0.5`` — the
    ``jnp.where`` would re-promote them to arrays, which the numeric halo
    tracer rejects.
    """
    if isinstance(coeff, int | float | complex):
        return coeff ** 0.5
    pos = coeff > 0
    return jnp.where(pos, jnp.where(pos, coeff, 1.0) ** 0.5, 0.0)


# ================================================================
#  The shared closure machinery
# ================================================================
class _DiffusionClosure(ClosureBase):

    """
    Shared machinery of the four diffusion/friction closures.

    Description
    -----------
    Concrete subclasses declare the class vocabulary (coefficient
    attribute/argument name, canonical parameter names, operator
    order, default targets) and a thin ``__init__``; everything else
    — coercion, bind-time axis resolution, the published parameters,
    and the tendency term — lives here.
    """

    _coeff_attr: ClassVar[str]
    _coeff_param: ClassVar[ParamName]
    _coeff_param_v: ClassVar[ParamName]
    _biharmonic: ClassVar[bool] = False
    _term_name: ClassVar[str]
    _units: ClassVar[str]
    _doc: ClassVar[str]

    def __init__(
        self,
        value: object,
        value_v: object,
        vertical: str,
        fields: Role | type[Role] | str | Iterable[str] | None,
        exclude: str | Iterable[str],
        slip: object = FREE_SLIP,
    ) -> None:
        """Coerce the coefficients and store the geometry names."""
        super().__init__(fields=fields, exclude=exclude)
        cls = type(self)
        owner = cls.__name__
        arg = cls._coeff_attr
        setattr(self, arg, _coerce_coefficient(
            value, arg, owner, nonnegative=cls._biharmonic))
        setattr(self, f"{arg}_v", None if value_v is None
                else _coerce_coefficient(
                    value_v, f"{arg}_v", owner,
                    nonnegative=cls._biharmonic))
        self._vertical = vertical
        self._slip = _coerce_slip(slip, owner)
        self._target_axes: tuple = ()
        self._no_slip_axes: dict[str, frozenset[str]] = {}

    # ================================================================
    #  Published parameters and per-field options
    # ================================================================
    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """The coefficient provides (the ``*_v`` one only if set)."""
        cls = type(self)
        declarations = (
            ParameterDeclaration(
                cls._coeff_param, attr=cls._coeff_attr,
                units=cls._units, doc=cls._doc),
        )
        if getattr(self, f"{cls._coeff_attr}_v") is not None:
            declarations += (
                ParameterDeclaration(
                    cls._coeff_param_v,
                    attr=f"{cls._coeff_attr}_v",
                    units=cls._units, doc=f"vertical {cls._doc}"),
            )
        return declarations

    @property
    def per_field_options(self) -> Mapping[str, object]:
        """The coefficient/slip slots, for the bind-time validation."""
        cls = type(self)
        options: dict[str, object] = {
            cls._coeff_attr: getattr(self, cls._coeff_attr)}
        value_v = getattr(self, f"{cls._coeff_attr}_v")
        if value_v is not None:
            options[f"{cls._coeff_attr}_v"] = value_v
        # a scalar slip passes through unchecked; a per-field mapping is
        # validated against the resolved targets (keys + full coverage)
        options["slip"] = self._slip
        return options

    # ================================================================
    #  Bind: per-target axes + wall treatment, vertical + slip check
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        """Resolve targets (base) and freeze the per-target axes.

        Each axis is classified into its wall treatment (periodic,
        wall-normal, or tangential/tracer), stored horizontal-then-
        vertical so the float-summation order matches the purely
        periodic path bit-for-bit.

        Raises
        ------
        NotImplementedError
            If a target factor carries an unsupported wall placement
            (a fixed-value cell wall or a bare face; see
            :func:`_wall_treatment`).
        AssemblyError
            If a ``*_v`` coefficient is given but no target carries
            the ``vertical`` coordinate, or a target has no
            coordinate axes at all.
        """
        super().bind(table)
        cls = type(self)
        owner = cls.__name__
        has_v = getattr(self, f"{cls._coeff_attr}_v")
        target_axes: list[
            tuple[str, tuple[tuple[str, bool, str], ...]]] = []
        for name in self.targets:
            space = table[name].space
            axes = tuple(space.names)
            if not axes:
                raise AssemblyError(
                    f"{owner}: target {name!r} has no coordinate "
                    "axes to diffuse along; exclude it "
                    "(exclude=...) or narrow fields=")
            if has_v is None:
                ordered = [(a, False) for a in axes]
            else:
                ordered = (
                    [(a, False) for a in axes if a != self._vertical]
                    + [(a, True) for a in axes if a == self._vertical])
            spec = tuple(
                (a, is_v, _wall_treatment(
                    space.bare.factor(a), owner, name, a))
                for a, is_v in ordered)
            target_axes.append((name, spec))
        if has_v is not None and not any(
                is_v for _, spec in target_axes
                for _, is_v, _ in spec):
            arg = cls._coeff_attr
            raise AssemblyError(
                f"{owner} got {arg}_v= but no target carries the "
                f"vertical coordinate {self._vertical!r}; drop "
                f"{arg}_v= or pass vertical=<coordinate name>")
        self._target_axes = tuple(target_axes)
        self._no_slip_axes = self._resolve_no_slip_axes()

    def _resolve_no_slip_axes(self) -> dict[str, frozenset[str]]:
        """Freeze the bounded tangential axes each target no-slips on.

        The base's :attr:`per_field_options` coverage check guarantees a
        per-field slip mapping keys (and covers) every target.
        """
        by_target = (
            self._slip if isinstance(self._slip, Mapping)
            else dict.fromkeys(self.targets, self._slip))
        return {
            name: frozenset(
                axis for axis, _is_v, treatment in spec
                if treatment == _TANGENTIAL)
            for name, spec in self._target_axes
            if by_target[name] == NO_SLIP}

    @staticmethod
    def _axes(
        spec: tuple[tuple[str, bool, str], ...],
        kh: object,
        kv: object,
    ) -> tuple[tuple[str, object, str], ...]:
        """Bind each axis to its coefficient (vertical picks ``kv``)."""
        return tuple(
            (axis, kv if is_v else kh, treatment)
            for axis, is_v, treatment in spec)

    # ================================================================
    #  The tendency term
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One linear term advancing every resolved target."""
        cls = type(self)
        return (
            TendencyTerm(
                name=cls._term_name, fn=self._apply,
                treatment=Treatment.EXPLICIT,
                advances=self.targets, linear=True),
        )

    def _apply(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Apply the (bi)harmonic operator to every target."""
        cls = type(self)
        coeff = ctx.params[cls._coeff_param]
        has_v = getattr(self, f"{cls._coeff_attr}_v") is not None
        coeff_v = ctx.params[cls._coeff_param_v] if has_v else None
        out: dict[str, ScalarField] = {}
        for name, spec in self._target_axes:
            q = state[name]
            kh = coeff[name] if isinstance(coeff, dict) else coeff
            kv = None
            if any(is_v for _, is_v, _ in spec):
                kv = (coeff_v[name] if isinstance(coeff_v, dict)
                      else coeff_v)
            no_slip = self._no_slip_axes.get(name, frozenset())
            if cls._biharmonic:
                # sqrt split, guarded for reverse-mode AD at coeff=0
                # (plain ** 0.5 keeps demoted Python scalars scalar; the
                # guard adds the jnp.where only for concrete jnp values)
                kh = _biharmonic_root(kh)
                kv = None if kv is None else _biharmonic_root(kv)
                axes = self._axes(spec, kh, kv)
                inner = _harmonic(q, axes, no_slip)
                out[name] = -_harmonic(inner, axes, no_slip)
            else:
                out[name] = _harmonic(
                    q, self._axes(spec, kh, kv), no_slip)
        return out


# ================================================================
#  The concrete closures
# ================================================================
@partial(jaxify, dynamic=("kappa", "kappa_v"))
class HarmonicDiffusion(_DiffusionClosure):

    r"""
    Harmonic mixing :math:`\nabla \cdot (\kappa \nabla q)` of tracers.

    Description
    -----------
    Targets every ``fr.roles.TRACER`` field by default; provides
    ``mixing.kappa`` (and ``mixing.kappa_v`` when given). On walled
    grids tracers get **no-flux** (zero wall flux) walls structurally
    — there is no slip choice (that is friction-only).

    Parameters
    ----------
    kappa : float | fr.Ramp | Mapping[str, float]
        Mixing coefficient (:math:`\mathrm{m^2/s}`): one scalar (or
        ramp) for every target, or a name-keyed per-field mapping of
        plain numbers covering every target.
    kappa_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical mixing coefficient, acting along ``vertical``;
        ``None`` applies ``kappa`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = TRACER
    _coeff_attr = "kappa"
    _coeff_param = MIXING_KAPPA
    _coeff_param_v = MIXING_KAPPA_V
    _term_name = "mixing"
    _units = "m^2/s"
    _doc = "harmonic mixing coefficient"

    def __init__(
        self,
        kappa: object,
        *,
        kappa_v: object = None,
        vertical: str = "z",
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the mixing coefficients; targets resolve at bind."""
        super().__init__(kappa, kappa_v, vertical, fields, exclude)


@partial(jaxify, dynamic=("kappa", "kappa_v"))
class BiharmonicDiffusion(_DiffusionClosure):

    r"""
    Biharmonic mixing :math:`-\mathcal{H}(\mathcal{H}(q))` of tracers.

    Description
    -----------
    The Griffies & Hallberg (2000) iterated form with square-rooted
    coefficients, so ``kappa`` is the true (non-negative) biharmonic
    coefficient in :math:`\mathrm{m^4/s}`. Targets every
    ``fr.roles.TRACER`` field by default; provides ``mixing.kappa4``
    (and ``mixing.kappa4_v`` when given). On walled grids tracers get
    **no-flux** walls on both Laplacian passes (no slip choice).

    Parameters
    ----------
    kappa : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic mixing coefficient
        (:math:`\mathrm{m^4/s}`); scalar, ramp, or per-field mapping.
    kappa_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic coefficient along ``vertical``; ``None``
        applies ``kappa`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = TRACER
    _coeff_attr = "kappa"
    _coeff_param = MIXING_KAPPA4
    _coeff_param_v = MIXING_KAPPA4_V
    _biharmonic = True
    _term_name = "mixing"
    _units = "m^4/s"
    _doc = "biharmonic mixing coefficient"

    def __init__(
        self,
        kappa: object,
        *,
        kappa_v: object = None,
        vertical: str = "z",
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the mixing coefficients; targets resolve at bind."""
        super().__init__(kappa, kappa_v, vertical, fields, exclude)


@partial(jaxify, dynamic=("nu", "nu_v"))
class HarmonicFriction(_DiffusionClosure):

    r"""
    Harmonic friction :math:`\nabla \cdot (\nu \nabla u_i)`.

    Description
    -----------
    Targets the ``Velocity`` role family by default (the PROGNOSTIC
    members, V-H2); provides ``friction.nu`` (and ``friction.nu_v``
    when given).

    On a walled grid the wall stress follows ``slip=``. The choice
    applies to every velocity factor that is **tangential** (nodal
    cell-centred, ``Center``) along a walled axis:
    ``slip="free"`` (the default) sets zero tangential wall stress —
    the structural zero wall flux, no spurious drag on a wall-parallel
    flow; ``slip="no"`` sets ``u = 0`` at the wall via the
    factor-of-2 ghost, adding the wall-adjacent drag
    :math:`-2\,\nu\,u_1/\Delta n^2`. The **wall-normal** component
    (staggered, ``Inner[Dirichlet]`` along the walled axis) ignores
    ``slip=`` — impermeability closes it either way. The default is a
    documented convention (MOM6 / Oceananigans default free-slip;
    MITgcm defaults no-slip — both are valid physics). Per-wall
    refinement is future work (the boundary-data path); a per-field
    mapping picks the slip per velocity.

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Viscosity (:math:`\mathrm{m^2/s}`); scalar, ramp, or
        name-keyed per-field mapping covering every target.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical viscosity along ``vertical``; ``None`` applies
        ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    slip : str | Mapping[str, str], optional
        Wall stress on tangential velocity factors: ``"free"`` (zero
        wall stress) or ``"no"`` (``u = 0`` at the wall). A scalar
        applies to every target; a name-keyed mapping (covering every
        target) picks per velocity. Static — a change recompiles
        (default: ``"free"``).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = Velocity
    _coeff_attr = "nu"
    _coeff_param = FRICTION_NU
    _coeff_param_v = FRICTION_NU_V
    _term_name = "friction"
    _units = "m^2/s"
    _doc = "harmonic friction coefficient (viscosity)"

    def __init__(
        self,
        nu: object,
        *,
        nu_v: object = None,
        vertical: str = "z",
        slip: str | Mapping[str, str] = FREE_SLIP,
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities and slip; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude, slip=slip)


@partial(jaxify, dynamic=("nu", "nu_v"))
class BiharmonicFriction(_DiffusionClosure):

    r"""
    Biharmonic friction :math:`-\mathcal{H}(\mathcal{H}(u_i))`.

    Description
    -----------
    The iterated square-rooted form (see `BiharmonicDiffusion`) on
    the ``Velocity`` role family (PROGNOSTIC members, V-H2);
    provides ``friction.nu4`` (and ``friction.nu4_v`` when given).

    On a walled grid the **same** slip treatment is applied on both
    Laplacian passes (Griffies & Hallberg); see `HarmonicFriction`
    for the ``slip=`` semantics.

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic viscosity (:math:`\mathrm{m^4/s}`);
        scalar, ramp, or per-field mapping.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic viscosity along ``vertical``; ``None``
        applies ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    slip : str | Mapping[str, str], optional
        Wall stress on tangential velocity factors, applied on both
        passes; ``"free"`` (default) or ``"no"`` (see
        `HarmonicFriction`).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = Velocity
    _coeff_attr = "nu"
    _coeff_param = FRICTION_NU4
    _coeff_param_v = FRICTION_NU4_V
    _biharmonic = True
    _term_name = "friction"
    _units = "m^4/s"
    _doc = "biharmonic friction coefficient"

    def __init__(
        self,
        nu: object,
        *,
        nu_v: object = None,
        vertical: str = "z",
        slip: str | Mapping[str, str] = FREE_SLIP,
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities and slip; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude, slip=slip)
