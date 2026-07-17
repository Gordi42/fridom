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

**Walled grids are future work**: the flux stencils next to rigid
walls (free-slip vs no-slip) are not covered, so ``bind`` rejects
walled grids with a taught error (the CenteredAdvection precedent).
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from fridom.framework.utils import jaxify
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
#  The shared harmonic pass (pure field arithmetic)
# ================================================================
def _harmonic(
    q: ScalarField,
    kh: object,
    kv: object,
    h_axes: tuple[str, ...],
    v_axes: tuple[str, ...],
) -> ScalarField:
    """One ``div(A grad q)`` pass with the coefficient in the flux."""
    res = None
    for axis in h_axes:
        contribution = (q.diff(axis) * kh).diff(axis)
        res = (contribution if res is None
               else res + contribution)
    for axis in v_axes:
        contribution = (q.diff(axis) * kv).diff(axis)
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
        self._target_axes: tuple = ()

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
        """The coefficient slots, for the bind-time key validation."""
        cls = type(self)
        options: dict[str, object] = {
            cls._coeff_attr: getattr(self, cls._coeff_attr)}
        value_v = getattr(self, f"{cls._coeff_attr}_v")
        if value_v is not None:
            options[f"{cls._coeff_attr}_v"] = value_v
        return options

    # ================================================================
    #  Bind: axes per target, wall rejection, vertical check
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        """Resolve targets (base) and freeze the per-target axes.

        Raises
        ------
        NotImplementedError
            On a walled grid (any bounded mesh factor): the flux
            stencils next to rigid walls (free-slip vs no-slip) are
            future work.
        AssemblyError
            If a ``*_v`` coefficient is given but no target carries
            the ``vertical`` coordinate, or a target has no
            coordinate axes at all.
        """
        super().bind(table)
        cls = type(self)
        owner = cls.__name__
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise NotImplementedError(
                f"{owner} does not support walled grids yet "
                f"(bounded coordinates: {walled}); the diffusive "
                "flux stencils next to rigid walls (free-slip vs "
                "no-slip) are future work — drop the closure on "
                "walled grids")
        has_v = getattr(self, f"{cls._coeff_attr}_v")
        target_axes: list[tuple[str, tuple[str, ...],
                                tuple[str, ...]]] = []
        for name in self.targets:
            axes = tuple(table[name].space.names)
            if not axes:
                raise AssemblyError(
                    f"{owner}: target {name!r} has no coordinate "
                    "axes to diffuse along; exclude it "
                    "(exclude=...) or narrow fields=")
            if has_v is None:
                h_axes, v_axes = axes, ()
            else:
                h_axes = tuple(a for a in axes
                               if a != self._vertical)
                v_axes = tuple(a for a in axes
                               if a == self._vertical)
            target_axes.append((name, h_axes, v_axes))
        if has_v is not None and not any(
                v_axes for _, _, v_axes in target_axes):
            arg = cls._coeff_attr
            raise AssemblyError(
                f"{owner} got {arg}_v= but no target carries the "
                f"vertical coordinate {self._vertical!r}; drop "
                f"{arg}_v= or pass vertical=<coordinate name>")
        self._target_axes = tuple(target_axes)

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
        for name, h_axes, v_axes in self._target_axes:
            q = state[name]
            kh = coeff[name] if isinstance(coeff, dict) else coeff
            kv = None
            if v_axes:
                kv = (coeff_v[name] if isinstance(coeff_v, dict)
                      else coeff_v)
            if cls._biharmonic:
                # sqrt split, guarded for reverse-mode AD at coeff=0
                # (plain ** 0.5 keeps demoted Python scalars scalar; the
                # guard adds the jnp.where only for concrete jnp values)
                kh = _biharmonic_root(kh)
                kv = None if kv is None else _biharmonic_root(kv)
                inner = _harmonic(q, kh, kv, h_axes, v_axes)
                out[name] = -_harmonic(inner, kh, kv,
                                       h_axes, v_axes)
            else:
                out[name] = _harmonic(q, kh, kv, h_axes, v_axes)
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
    ``mixing.kappa`` (and ``mixing.kappa_v`` when given).

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
        The vertical coordinate name (default: ``"z"``).
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
    (and ``mixing.kappa4_v`` when given).

    Parameters
    ----------
    kappa : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic mixing coefficient
        (:math:`\mathrm{m^4/s}`); scalar, ramp, or per-field mapping.
    kappa_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic coefficient along ``vertical``; ``None``
        applies ``kappa`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
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

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Viscosity (:math:`\mathrm{m^2/s}`); scalar, ramp, or
        name-keyed per-field mapping covering every target.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical viscosity along ``vertical``; ``None`` applies
        ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
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
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude)


@partial(jaxify, dynamic=("nu", "nu_v"))
class BiharmonicFriction(_DiffusionClosure):

    r"""
    Biharmonic friction :math:`-\mathcal{H}(\mathcal{H}(u_i))`.

    Description
    -----------
    The iterated square-rooted form (see `BiharmonicDiffusion`) on
    the ``Velocity`` role family (PROGNOSTIC members, V-H2);
    provides ``friction.nu4`` (and ``friction.nu4_v`` when given).

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic viscosity (:math:`\mathrm{m^4/s}`);
        scalar, ramp, or per-field mapping.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic viscosity along ``vertical``; ``None``
        applies ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
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
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude)
