r"""
Unit factors and scales report (``model.units``).

Description
-----------
The §D surface of the nondimensionalization plan
(``design/plans/active/nondimensionalization_plan.md``): the model
exposes the **dimensional conversion factors** of its components,
coordinates and time axis — it never converts state itself. Every
factor is derivable from the scaling object's stored reference
scales (``L``, ``U`` [, ``g``]) plus live bound parameters, through
the one rule :math:`T_\mathrm{ref} = \varepsilon\,L/U` — no
per-scaling case analysis, no copies that can go stale under
sweeps/ramps (everything is recomputed live on every access).

Modules contribute rows through a duck-typed ``unit_factors``
mapping attribute (``{name: UnitFactor}`` — the ``diagnostics``
channel precedent); the framework itself contributes the ``t`` and
``T_ref`` rows. On a dimensional model (``fr.scaling.Dimensional()``
or ``scaling=None``) the raw component/coordinate/time rows are
identity (1.0 with the physical unit strings — scripts stay
polymorphic), while **curated** rows keep their meaning (e.g. the
shallow-water ``factor("h")`` is ``1/g`` from the bound gravity, so
``factor("h") * p`` yields meters in both variants) and derived
**constants** report the bound physical values where the provides
exist.

Only :meth:`UnitsView.factor` raises (taught errors naming the
fix); :attr:`UnitsView.factors` and :meth:`UnitsView.report` mark
unresolvable rows instead.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fridom.model.errors import AssemblyError
from fridom.model.params import SCALING_NONLINEARITY
from fridom.model.time_dependent import TimeDependent, resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

#: the raw row kinds — identity (1.0) on a dimensional model
_RAW_KINDS = frozenset({"component", "coordinate", "time"})

#: the full row-kind vocabulary
_KINDS = _RAW_KINDS | {"curated", "constant"}


# ================================================================
#  UnitFactor (the declared row) and FactorEntry (one resolution)
# ================================================================
@dataclass(frozen=True)
class UnitFactor:

    r"""
    One declared dimensional-factor row (frozen, host-side).

    Description
    -----------
    Describes how one factor is computed from the scaling object's
    stored reference scales and live bound parameters. ``fn``
    receives one mapping of resolved symbol values (the ``scales``
    names plus the ``params`` keys) and returns the factor. The
    ``dim_*`` triple is the **dimensional-model** resolution of
    curated/constant rows (raw rows are identity there and never
    read it): ``dim_params`` name the bound physical constants and
    ``dim_fn`` combines them (``None`` falls back to identity).

    Parameters
    ----------
    unit : str
        The target physical unit string (e.g. ``"m/s"``).
    expr : str
        The nondimensional factor expression, documentation only
        (e.g. ``"U^2/(eps*g)"``).
    kind : str
        One of ``component`` / ``coordinate`` / ``time`` (raw rows)
        or ``curated`` / ``constant``.
    scales : tuple[str, ...], optional
        The scaling-object reference scales the factor reads
        (``"L"`` / ``"U"`` / ``"g"``) (default: ()).
    params : Mapping[str, str], optional
        Symbol -> dotted bound-parameter name (``ParamName``) reads
        (default: {}).
    fn : Callable | None, optional
        ``fn(values) -> float`` over the resolved symbols
        (default: None).
    dim_expr : str, optional
        The dimensional-model expression (curated/constant rows)
        (default: "1").
    dim_params : Mapping[str, str], optional
        The dimensional-model bound-constant reads (default: {}).
    dim_fn : Callable | None, optional
        The dimensional-model resolver; ``None`` is identity
        (default: None).
    """

    unit: str
    expr: str
    kind: str
    scales: tuple[str, ...] = ()
    params: Mapping[str, str] = field(default_factory=dict)
    fn: Callable[[Mapping[str, float]], float] | None = None
    dim_expr: str = "1"
    dim_params: Mapping[str, str] = field(default_factory=dict)
    dim_fn: Callable[[Mapping[str, float]], float] | None = None

    def __post_init__(self) -> None:
        """Validate the row kind (taught at declaration time)."""
        if self.kind not in _KINDS:
            kinds = ", ".join(sorted(_KINDS))
            raise ValueError(
                f"unknown unit-factor kind {self.kind!r}; kinds "
                f"are {kinds}")


@dataclass(frozen=True)
class FactorEntry:

    """
    One resolved factor row (a live snapshot, never cached).

    Description
    -----------
    ``value`` is ``None`` when the row cannot be resolved:
    ``missing`` then lists the unset reference scales (as ``"U="``
    marks) and/or unbound parameter names (dotted ``ParamName``
    strings), and ``time_dependent`` marks a Ramp-valued input that
    needs an explicit ``at=`` (D2.4: no implicit clock evaluation).

    Parameters
    ----------
    name : str
        The row name.
    value : float | None
        The resolved factor, or ``None`` (see above).
    unit : str
        The target physical unit string.
    kind : str
        The row kind (see :class:`UnitFactor`).
    expr : str
        The active variant's expression (``"1"`` for identity).
    missing : tuple[str, ...], optional
        Unset scales / unbound parameter names (default: ()).
    time_dependent : bool, optional
        Whether a Ramp-valued input needs ``at=`` (default: False).
    """

    name: str
    value: float | None
    unit: str
    kind: str
    expr: str
    missing: tuple[str, ...] = ()
    time_dependent: bool = False


# ================================================================
#  The framework rows: t and T_ref (the one rule, eps*L/U)
# ================================================================
def _t_ref(values: Mapping[str, float]) -> float:
    """Apply the one rule ``T_ref = eps * L / U``."""
    return values["eps"] * values["L"] / values["U"]


def _identity(values: Mapping[str, float]) -> float:  # noqa: ARG001
    """Return the identity factor (dimensional seconds)."""
    return 1.0


_TIME_FACTOR = UnitFactor(
    unit="s", expr="eps*L/U", kind="time", scales=("L", "U"),
    params={"eps": SCALING_NONLINEARITY}, fn=_t_ref)

_T_REF_FACTOR = UnitFactor(
    unit="s", expr="eps*L/U", kind="constant", scales=("L", "U"),
    params={"eps": SCALING_NONLINEARITY}, fn=_t_ref,
    dim_expr="1", dim_fn=_identity)

#: display order of the row kinds in :meth:`UnitsView.report`
_REPORT_ORDER = ("constant", "time", "coordinate", "component",
                 "curated")


# ================================================================
#  UnitsView (the model.units surface)
# ================================================================
class UnitsView:

    """
    ``model.units`` — live dimensional factors and scales report.

    Description
    -----------
    A host-side read view (never a pytree, never traced): every
    access re-collects the module rows and re-resolves them against
    the current scaling object and live bound parameters, so the
    factors stay correct under parameter sweeps and ramps. Ramp
    inputs are never silently evaluated — pass ``at=`` (D2.4).

    Parameters
    ----------
    model : Model
        The owning model (duck-typed: ``modules``, ``scaling``,
        ``parameters``).
    """

    def __init__(self, model: object) -> None:
        """Bind the view to the model; see the class docstring."""
        self._model = model

    # ================================================================
    #  Collection (the diagnostics per-name-collision pattern)
    # ================================================================
    def _entries(self) -> dict[str, UnitFactor]:
        """Collect the framework + module rows (live, collision-checked).

        Raises
        ------
        AssemblyError
            If two contributors declare the same row name.
        """
        entries: dict[str, UnitFactor] = {
            "t": _TIME_FACTOR, "T_ref": _T_REF_FACTOR}
        owners = dict.fromkeys(entries, "the framework")
        for module in self._model.modules:
            mapping = getattr(module, "unit_factors", None)
            if not isinstance(mapping, Mapping):
                continue
            owner = type(module).__name__
            for name, factor in mapping.items():
                if name in entries:
                    raise AssemblyError(
                        f"unit factor {name!r} is contributed by "
                        f"both {owners[name]} and {owner}; "
                        "unit-factor names are unique per model")
                entries[name] = factor
                owners[name] = owner
        return entries

    # ================================================================
    #  Resolution (live; the centralized variant switch)
    # ================================================================
    def _resolve(
        self, name: str, factor: UnitFactor, *,
        at: float | None = None,
    ) -> FactorEntry:
        """Resolve one row against the scaling + live parameters."""
        scaling = self._model.scaling
        nondimensional = bool(
            getattr(scaling, "nondimensional", False))
        if nondimensional:
            expr, scales = factor.expr, factor.scales
            params, fn = factor.params, factor.fn
        elif factor.kind in _RAW_KINDS or factor.dim_fn is None:
            # dimensional model: raw rows are identity (physical
            # unit strings kept — scripts stay polymorphic)
            return FactorEntry(name=name, value=1.0,
                               unit=factor.unit, kind=factor.kind,
                               expr="1")
        else:
            # curated/constant rows keep their meaning: resolve the
            # bound physical constants (owner ruling 2026-07-21)
            expr, scales = factor.dim_expr, ()
            params, fn = factor.dim_params, factor.dim_fn
        missing: list[str] = []
        values: dict[str, float] = {}
        for scale in scales:
            stored = getattr(scaling, scale, None)
            if stored is None:
                missing.append(f"{scale}=")
            else:
                values[scale] = float(stored)
        time_dependent = False
        parameters = self._model.parameters
        for symbol, param_name in params.items():
            if param_name not in parameters:
                missing.append(param_name)
                continue
            value = parameters[param_name]
            if isinstance(value, TimeDependent):
                if at is None:
                    time_dependent = True
                    continue
                value = resolve_at(value, at)
            values[symbol] = float(value)
        if missing or time_dependent:
            return FactorEntry(
                name=name, value=None, unit=factor.unit,
                kind=factor.kind, expr=expr,
                missing=tuple(missing),
                time_dependent=time_dependent)
        return FactorEntry(name=name, value=float(fn(values)),
                           unit=factor.unit, kind=factor.kind,
                           expr=expr)

    # ================================================================
    #  The public surface
    # ================================================================
    @property
    def factors(self) -> dict[str, FactorEntry]:
        """All rows resolved live; unresolvable rows are marked.

        Description
        -----------
        Never raises on unresolved inputs: a row whose scales or
        parameters are unavailable carries ``value=None`` with its
        ``missing`` / ``time_dependent`` marks (see
        :class:`FactorEntry`).
        """
        return {name: self._resolve(name, factor)
                for name, factor in self._entries().items()}

    def factor(self, name: str, *, at: float | None = None) -> float:
        """
        Return one resolved factor (taught errors on failure).

        Parameters
        ----------
        name : str
            The row name (a component, coordinate, ``"t"``, a
            curated name, or a derived constant).
        at : float | None, optional
            The model time at which Ramp-valued inputs are
            evaluated; required when one is a Ramp (D2.4)
            (default: None).

        Returns
        -------
        float
            The dimensional factor.

        Raises
        ------
        ValueError
            Unknown name, unset reference scale, unbound referenced
            parameter, or a Ramp input without ``at=`` — each named
            with its fix.
        """
        entries = self._entries()
        if name not in entries:
            available = ", ".join(sorted(entries))
            raise ValueError(
                f"no unit factor named {name!r}; available "
                f"factors: {available}")
        entry = self._resolve(name, entries[name], at=at)
        if entry.value is not None:
            return entry.value
        problems: list[str] = []
        for item in entry.missing:
            if item.endswith("="):
                problems.append(
                    f"the reference scale {item[:-1]} is unset — "
                    f"pass {item}<value> on the model's fr.scaling "
                    "object")
            else:
                hint = getattr(item, "hint", "")
                problems.append(
                    f"the referenced parameter {item!r} is not "
                    "bound on this model"
                    + (f" ({hint})" if hint else ""))
        if entry.time_dependent:
            problems.append(
                "a referenced parameter is time-dependent (a "
                "Ramp); pass at=<model time> to evaluate it, e.g. "
                f"units.factor({name!r}, at=0.0)")
        raise ValueError(
            f"unit factor {name!r} cannot be resolved: "
            + "; ".join(problems))

    def bound_constants(self) -> dict[str, float]:
        """
        Collect the bound constants the active rows reference.

        Description
        -----------
        The writer's ``fridom_scaling_parameters`` stamp source:
        every dotted parameter name referenced by a row (in the
        model's active variant) that is bound to a plain constant.
        Ramp-valued and unbound names are skipped (best-effort
        provenance, never an error).

        Returns
        -------
        dict[str, float]
            Dotted name -> bound constant value.
        """
        scaling = self._model.scaling
        nondimensional = bool(
            getattr(scaling, "nondimensional", False))
        parameters = self._model.parameters
        constants: dict[str, float] = {}
        for factor in self._entries().values():
            params = (factor.params if nondimensional
                      else factor.dim_params)
            for param_name in params.values():
                key = str(param_name)
                if key in constants or param_name not in parameters:
                    continue
                value = parameters[param_name]
                if isinstance(value, TimeDependent):
                    continue
                constants[key] = float(value)
        return constants

    def report(self, *, at: float | None = None) -> str:
        """
        Human-readable scales + factors report (never raises).

        Description
        -----------
        Prints the scaling header (class, variant, stored scales),
        then every row grouped by kind; rows the stored scales
        cannot resolve are marked ``-- needs U=`` (etc.), and
        Ramp-valued rows ``-- time-dependent (pass at=)``.

        Parameters
        ----------
        at : float | None, optional
            The model time for Ramp-valued inputs (default: None).

        Returns
        -------
        str
            The formatted report.
        """
        scaling = self._model.scaling
        nondimensional = bool(
            getattr(scaling, "nondimensional", False))
        label = (type(scaling).__name__ if scaling is not None
                 else "no scaling")
        variant = ("nondimensional" if nondimensional
                   else "dimensional")
        lines = [f"unit factors ({label}, {variant})"]
        if scaling is not None:
            stored = ", ".join(
                f"{name} = "
                + ("unset" if getattr(scaling, name, None) is None
                   else f"{float(getattr(scaling, name)):g}")
                for name in ("L", "U", "g"))
            lines.append(f"  scales: {stored}")
        entries = self._entries()
        width = max(len(name) for name in entries)
        ordered = sorted(
            entries, key=lambda name: (
                _REPORT_ORDER.index(entries[name].kind), name))
        for name in ordered:
            entry = self._resolve(name, entries[name], at=at)
            if entry.value is not None:
                body = f"= {entry.value:.6g} {entry.unit}"
            elif entry.missing:
                needs = ", ".join(str(m) for m in entry.missing)
                body = f"-- needs {needs}"
            else:
                body = "-- time-dependent (pass at=)"
            lines.append(
                f"  {name:<{width}} {body}  [{entry.expr}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Summary repr listing the row names."""
        names = ", ".join(sorted(self._entries()))
        return f"<units: {names}>"
