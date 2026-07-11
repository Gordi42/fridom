r"""
Shared relaxation (nudging) module: restore fields toward targets.

Description
-----------
The framework's reusable relaxation forcing (the framework2 port of
the v1 ``fr.modules.forcings.Relaxation``): one or more PROGNOSTIC
fields are nudged toward target profiles

.. math::
    \partial_t \phi \leftarrow \partial_t \phi
        + r\,M(\boldsymbol{x})\,(\phi^\ast - \phi)

with the relaxation rate :math:`r = 1/\tau` (the v1 module took the
time scale :math:`\tau`), the target profile :math:`\phi^\ast`, and
an optional stationary mask :math:`M` replacing the v1 boolean
``domain_function`` (0/1 values reproduce it; smooth ramps avoid the
hard sponge edge).

Following R2 (01_concepts D2.2), the target and mask are
*intrinsically spatial* and are carried as AUXILIARY fields on the
smallest ``fr.Profile`` that represents them: a constant target is a
one-DOF ``fr.Profile()``; a callable target/mask varies exactly
along its named coordinates (``lambda z: ...`` -> ``fr.Profile("z")``)
and is sampled at those coordinates' nodes when the field is
materialized (the ``MeridionalStratification`` precedent). In the
term the profile is moved onto each relaxed field with ``.to`` — a
pure broadcast on shared nodes, the measure-weighted interpolation
onto staggered faces — so the term is pure field arithmetic, traced
normally, with no ``extra_halo``.

The rate is a provided dynamic-leaf parameter named
``relaxation.<fields>.rate``, so ``model.update_parameters`` sweeps
it without re-assembly and several `Relaxation` instances coexist.
"""
from __future__ import annotations

import inspect
import numbers
from functools import partial
from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import (
    ParameterDeclaration,
    leaf,
)
from fridom.framework2.model.params import ParamName
from fridom.framework2.model.space_patterns import Profile
from fridom.framework2.model.terms import TendencyTerm, Treatment

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

    from fridom.framework2.model.time_dependent import TimeDependent

_FIELD_HINT = ("the relaxed field must be declared by another "
               "module (e.g. a dynamical core registers the "
               "velocities, a stratification module registers b); "
               "check the Relaxation field-name spelling")


def _coordinate_names(fn: Callable) -> tuple[str, ...]:
    """Return the coordinate names a profile callable varies along."""
    return tuple(inspect.signature(fn).parameters)


def _normalize_fields(field: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize the relaxed-field selection to a name tuple."""
    names = (field,) if isinstance(field, str) else tuple(field)
    if not names:
        raise ValueError(
            "Relaxation needs at least one field name to relax")
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"Relaxation field names are non-empty strings, "
                f"got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(
            f"Relaxation field names must be unique, got {names}")
    return names


def _normalize_targets(
    fields: tuple[str, ...],
    target: object,
) -> dict[str, float | Callable]:
    """Normalize the target slot to one value per relaxed field."""
    if isinstance(target, dict):
        if set(target) != set(fields):
            raise ValueError(
                f"a mapping-valued target= keys every relaxed field "
                f"exactly once; relaxing {fields}, got keys "
                f"{tuple(sorted(target))}")
        values = {name: target[name] for name in fields}
    else:
        values = dict.fromkeys(fields, target)
    for name, value in values.items():
        if not (isinstance(value, numbers.Number)
                or callable(value)):
            raise TypeError(
                f"the target of {name!r} must be a number or a "
                f"profile callable of coordinate names, got "
                f"{value!r}")
    return values


@partial(jaxify, dynamic=("rate",))
class Relaxation(Module):

    r"""
    Relax (nudge) fields toward target profiles at a rate.

    Description
    -----------
    Contributes the tendency term

    .. math::
        \partial_t \phi \leftarrow \partial_t \phi
            + r\,M(\boldsymbol{x})\,(\phi^\ast - \phi)

    for every named field :math:`\phi`: exponential restoring toward
    the target :math:`\phi^\ast` at the rate :math:`r = 1/\tau`,
    optionally gated by a stationary mask :math:`M` (the v1
    ``domain_function``, as a smooth or 0/1 profile). Targets and
    the mask are AUXILIARY profile fields materialized at assembly:
    a number is a constant profile, a callable is sampled at the
    nodes of the coordinates its signature names
    (``lambda z: ...`` varies in z, constant elsewhere) and is
    interpolated onto each relaxed field with ``.to``.

    The rate is provided as the dynamic-leaf parameter
    ``relaxation.<fields>.rate``, so
    ``model.update_parameters({"relaxation.b.rate": ...})`` sweeps
    it without re-assembly.

    Parameters
    ----------
    field : str | Iterable[str]
        The PROGNOSTIC field name(s) to relax; unknown names raise a
        taught assembly error.
    rate : float | fr.Ramp, optional
        The relaxation rate :math:`r = 1/\tau` shared by all named
        fields; may be an ``fr.Ramp`` for a spun-up forcing
        (default: 1.0).
    target : float | Callable | Mapping[str, float | Callable], optional
        The target profile: a number, a callable of coordinate
        names, or a mapping with one entry per relaxed field
        (default: 0.0).
    mask : Callable | None, optional
        Stationary relaxation mask :math:`M` as a callable of
        coordinate names; None relaxes everywhere (default: None).
    """

    def __init__(
        self,
        field: str | Iterable[str],
        *,
        rate: float | TimeDependent = 1.0,
        target: object = 0.0,
        mask: Callable | None = None,
    ) -> None:
        """Store the rate leaf; freeze names, targets and the mask."""
        names = _normalize_fields(field)
        self._fields: tuple[str, ...] = names
        self.rate = leaf(rate)
        self._targets: dict[str, float | Callable] = (
            _normalize_targets(names, target))
        if mask is not None and not callable(mask):
            raise TypeError(
                f"mask= must be a profile callable of coordinate "
                f"names (or None to relax everywhere), got {mask!r}")
        joined = "_".join(names)
        self._mask_fn: Callable | None = mask
        self._mask_name: str | None = (
            None if mask is None else f"{joined}_relax_mask")
        self._rate_name: ParamName = ParamName(
            f"relaxation.{joined}.rate", units="1/s",
            hint="provided by the fr.modules.Relaxation instance "
                 f"relaxing {names}")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def fields(self) -> tuple[str, ...]:
        """The relaxed field names."""
        return self._fields

    @property
    def rate_parameter(self) -> ParamName:
        """The dotted name of the provided rate parameter."""
        return self._rate_name

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """One checked claim per relaxed field."""
        return tuple(FieldReference(name, hint=_FIELD_HINT)
                     for name in self._fields)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The AUXILIARY target profiles (plus the optional mask)."""
        declarations = [
            self._profile_declaration(
                f"{name}_relax_target", self._targets[name],
                long_name=f"Relaxation target of {name}")
            for name in self._fields]
        if self._mask_fn is not None:
            declarations.append(self._profile_declaration(
                self._mask_name, self._mask_fn,
                long_name="Relaxation mask", units="1"))
        return tuple(declarations)

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """Publish the live rate leaf under the instance's name."""
        return (
            ParameterDeclaration(
                self._rate_name, attr="rate", units="1/s",
                doc="relaxation rate 1/tau of "
                    f"{', '.join(self._fields)}"),
        )

    @staticmethod
    def _profile_declaration(
        name: str,
        value: float | Callable,
        *,
        long_name: str,
        units: str = "n/a",
    ) -> FieldDeclaration:
        """Declare one value as an AUXILIARY profile field.

        A number becomes a constant fill on the one-DOF
        ``fr.Profile()``; a callable becomes the coordinate default
        of a ``fr.Profile`` over exactly the coordinates its
        signature names (sampled at those nodes — the
        MeridionalStratification precedent).
        """
        if callable(value):
            space = Profile(*_coordinate_names(value))
            default: float | Callable = value
        else:
            space = Profile()
            default = float(value)
        return FieldDeclaration(
            name, space=space, lifecycle=Lifecycle.AUXILIARY,
            default=default, long_name=long_name, units=units)

    # ================================================================
    #  Bind-time validation
    # ================================================================
    def bind(self, table: object) -> None:
        """Check lifecycles and profile coordinate names.

        Raises
        ------
        ValueError
            If a relaxed field is not PROGNOSTIC (only stepper-
            advanced fields take tendency contributions), or if a
            target/mask callable names a coordinate the grid does
            not have.
        """
        for name in self._fields:
            lifecycle = table[name].lifecycle
            if lifecycle is not Lifecycle.PROGNOSTIC:
                raise ValueError(
                    f"Relaxation adds a tendency to {name!r}, which "
                    f"is {lifecycle.name}: only PROGNOSTIC fields "
                    "are advanced from tendencies — relax a "
                    "prognostic field instead")
        profiles = [(f"target of {name!r}", value)
                    for name, value in self._targets.items()
                    if callable(value)]
        if self._mask_fn is not None:
            profiles.append(("mask", self._mask_fn))
        grid_names = set(table.grid.names)
        for label, fn in profiles:
            unknown = sorted(set(_coordinate_names(fn)) - grid_names)
            if unknown:
                raise ValueError(
                    f"the Relaxation {label} callable names the "
                    f"coordinate(s) {unknown}, which the grid does "
                    f"not have (coordinates: {table.grid.names}); "
                    "profile callables vary along the coordinates "
                    "their parameters name")

    # ================================================================
    #  The relaxation term
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One term restoring every relaxed field."""
        return (
            TendencyTerm(
                name="relaxation", fn=self._relax,
                treatment=Treatment.EXPLICIT,
                advances=self._fields),
        )

    def _relax(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``d phi/dt += r M (phi* - phi)`` for every relaxed field.

        The target (and mask) profiles are moved onto each field's
        own space with ``.to`` — a pure broadcast on shared nodes,
        an interpolation onto staggered faces — so the term is pure
        field arithmetic and needs no ``extra_halo``.
        """
        rate = ctx.params[self._rate_name]
        out = {}
        for name in self._fields:
            f = state[name]
            delta = state[f"{name}_relax_target"].to(f) - f
            if self._mask_name is not None:
                delta = state[self._mask_name].to(f) * delta
            out[name] = rate * delta
        return out
