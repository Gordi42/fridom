"""
The closure base class (``fr.closures.ClosureBase``).

Description
-----------
The framework closure base (D1.4, V-H2): the ``fr.terms.owned_by``
predicate target that hosts the role-target resolution boilerplate
shared by every dissipative closure. Owning class spec:
``design/specs/model/classes/module.md`` (section
"fr.closures.ClosureBase").

Closures default their target set *by role* — mixing closures declare
``default_targets = fr.roles.TRACER``, friction closures the
``Velocity`` family (the class: family match) — and take name-keyed
constructor overrides (``fields=`` / ``exclude=`` / per-field
coefficient mappings), validated at bind. This replaces the old
``ENABLE_FRICTION`` / ``ENABLE_MIXING`` declaration-side flags.
Role-driven write-targeting intersects PROGNOSTIC automatically
(V-H2): ``Velocity`` may sit on DIAGNOSTIC fields (the hydrostatic
diagnosed ``w``), but a closure's write targets are PROGNOSTIC by
construction. Dropping every closure from a model is
``model.variant(term_filter=~fr.terms.owned_by(fr.closures.
ClosureBase))`` — the predicate follows free from subclassing.
"""
from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar

from fridom.model.declarations import Lifecycle
from fridom.model.errors import (
    AssemblyError,
    MissingFieldError,
)
from fridom.model.module import Module
from fridom.model.roles import Role

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.field_table import FieldTable


# ================================================================
#  Constructor-argument normalization helpers
# ================================================================
def _normalize_fields(
    fields: Role | type[Role] | str | Iterable[str] | None,
    owner: str,
) -> Role | type[Role] | tuple[str, ...] | None:
    """Normalize the ``fields=`` override; see `ClosureBase`."""
    if fields is None or isinstance(fields, Role):
        return fields
    if isinstance(fields, type):
        if issubclass(fields, Role):
            return fields
        raise TypeError(
            f"{owner}: fields= takes a role, a Role family class, "
            f"or field names; got the class {fields!r}")
    names = _normalize_names(fields, "fields", owner)
    if not names:
        raise ValueError(
            f"{owner}: fields=() selects nothing; omit fields= for "
            "the role-driven default targets")
    return names


def _normalize_names(
    names: str | Iterable[str], option: str, owner: str,
) -> tuple[str, ...]:
    """Coerce a name selection to a validated tuple of strings."""
    if isinstance(names, str):
        names = (names,)
    try:
        normalized = tuple(names)
    except TypeError:
        raise TypeError(
            f"{owner}: {option}= takes field names, got "
            f"{names!r}") from None
    for name in normalized:
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{owner}: {option}= entries are non-empty field "
                f"names, got {name!r}")
    return normalized


# ================================================================
#  ClosureBase
# ================================================================
class ClosureBase(Module, ABC):

    """
    Base for dissipative closures; hosts role-target defaults.

    Description
    -----------
    An abstract `Module` marker + helper with no hooks of its own:
    concrete closures contribute their tendency terms/stages through
    the normal Module capability menu. What the base earns its keep
    with is D1.4's target resolution: ``bind(table)`` resolves
    ``fields or default_targets`` minus ``exclude`` into the frozen
    :attr:`targets` tuple (table declaration order), intersected with
    PROGNOSTIC lifecycles (V-H2 — closures never write DIAGNOSTIC
    fields), and validates every name-keyed per-field option
    (:attr:`per_field_options`) against the resolved targets. Unknown
    names in ``fields=``, ``exclude=``, or a per-field coefficient
    mapping are assembly errors with taught messages.

    Subclassing also enrolls the closure in the
    ``fr.terms.owned_by(fr.closures.ClosureBase)`` predicate, so
    inviscid variants drop all closures with no new vocabulary.

    Parameters
    ----------
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override: a role instance (exact match), a Role family
        class (family match), or explicit field names. ``None`` uses
        the subclass's :attr:`default_targets` (default: None).
    exclude : str | Iterable[str], optional
        Field names removed from the resolved targets (default: ()).
    """

    default_targets: ClassVar[Role | type[Role]]
    """Subclass-declared default target selection: mixing closures
    set ``fr.roles.TRACER``, friction closures the ``Velocity``
    family (class = family match, D1.4)."""

    _allow_empty_targets: ClassVar[bool] = False
    """Subclass opt-out of the zero-target assembly error, for
    closures that do useful work beyond their role targets (e.g. a
    Smagorinsky closure whose stress term is target-independent)."""

    def __init__(
        self,
        *,
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the target override and exclusions (resolved at bind)."""
        owner = type(self).__name__
        self._fields = _normalize_fields(fields, owner)
        self._exclude = _normalize_names(exclude, "exclude", owner)
        self._targets: tuple[str, ...] = ()

    # ================================================================
    #  Bind-time target resolution (assembly step 4)
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        """
        Resolve the target selection into the frozen ``targets``.

        Description
        -----------
        Resolves ``fields or default_targets`` minus ``exclude`` into
        the static :attr:`targets` tuple (table declaration order),
        intersected with PROGNOSTIC lifecycles, then validates the
        name-keyed :attr:`per_field_options` mappings. Cooperative:
        subclasses call ``super().bind(table)`` first and read
        ``self.targets`` afterwards.

        Parameters
        ----------
        table : FieldTable
            The resolved field table.

        Raises
        ------
        MissingFieldError
            If ``fields=`` or ``exclude=`` names an undeclared field.
        AssemblyError
            If an explicit ``fields=`` name is not PROGNOSTIC (V-H2),
            the resolution is empty, the subclass declares no
            ``default_targets``, or a per-field option mapping keys
            an unknown target.
        """
        super().bind(table)
        owner = type(self).__name__
        if getattr(table.grid, "immersed", None) is not None:
            raise NotImplementedError(
                f"{owner} does not support immersed (cut-cell) grids: "
                "its strain / stress / diffusive-flux stencils next to "
                "the immersed boundary would read across dry cells "
                "unmasked, and fraction-weighting a viscous closure is "
                "not the mechanical edit the advective flux form is "
                "(immersed-partial-cells plan, IP-D8) — designed-for. "
                "Drop the closure on an immersed grid.")
        selection = self._fields
        if selection is None:
            selection = getattr(type(self), "default_targets", None)
            if selection is None:
                raise AssemblyError(
                    f"{owner} declares no default_targets and got "
                    "no fields=; closure subclasses declare a Role "
                    "(or Role family) default, e.g. default_targets"
                    " = fr.roles.TRACER")
        if isinstance(selection, Role) or (
                isinstance(selection, type)
                and issubclass(selection, Role)):
            # role-driven: silently intersect PROGNOSTIC (V-H2)
            names = [
                name for name in table.select(selection)
                if table[name].lifecycle is Lifecycle.PROGNOSTIC]
        else:
            names = self._resolve_explicit(selection, table, owner)
        for name in self._exclude:
            if name not in table:
                declared = ", ".join(table.names) or "none"
                raise MissingFieldError(
                    f"{owner}: unknown field {name!r} in exclude=; "
                    f"declared fields: {declared}")
        excluded = frozenset(self._exclude)
        targets = tuple(
            name for name in names if name not in excluded)
        if not targets and not self._allow_empty_targets:
            raise AssemblyError(
                f"{owner} resolves zero target fields (selection "
                f"{selection!r}, exclude {self._exclude!r}); pass "
                "fields=, adjust exclude=, or drop the closure "
                "from the module list")
        self._targets = targets
        self._validate_per_field_options()

    def _resolve_explicit(
        self,
        selection: tuple[str, ...],
        table: FieldTable,
        owner: str,
    ) -> list[str]:
        """Validate explicit names; return them in table order."""
        for name in selection:
            if name not in table:
                declared = ", ".join(table.names) or "none"
                raise MissingFieldError(
                    f"{owner}: unknown field {name!r} in fields=; "
                    f"declared fields: {declared}")
            lifecycle = table[name].lifecycle
            if lifecycle is not Lifecycle.PROGNOSTIC:
                raise AssemblyError(
                    f"{owner}: fields= names {name!r}, a "
                    f"{lifecycle.name} field; closures write "
                    "PROGNOSTIC fields only (V-H2 — a diagnosed "
                    "field has no tendency equation)")
        chosen = frozenset(selection)
        return [name for name in table.names if name in chosen]

    def _validate_per_field_options(self) -> None:
        """Check every per-field option mapping against the targets."""
        owner = type(self).__name__
        for option, value in self.per_field_options.items():
            if not isinstance(value, Mapping):
                continue
            unknown = tuple(
                key for key in value if key not in self._targets)
            if unknown:
                raise AssemblyError(
                    f"unknown target {unknown[0]!r} in {option}= of "
                    f"{owner}; resolved targets: {self._targets}")
            missing = tuple(
                name for name in self._targets if name not in value)
            if missing:
                raise AssemblyError(
                    f"{option}= of {owner} gives no coefficient for "
                    f"target {missing[0]!r} (resolved targets: "
                    f"{self._targets}); cover every target, or "
                    "narrow the targets with fields=/exclude=")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def targets(self) -> tuple[str, ...]:
        """Frozen post-bind target names, declaration order."""
        return self._targets

    @property
    def per_field_options(self) -> Mapping[str, object]:
        """
        Name-keyed per-field options validated at bind.

        Description
        -----------
        Subclasses expose their coefficient slots here (option name
        mapped to the stored value); every `Mapping`-valued entry is
        validated against the resolved targets at bind — unknown or
        uncovered target names are assembly errors. Scalar-valued
        entries pass through unchecked. Default: no options.

        Returns
        -------
        Mapping[str, object]
            The option name to stored value mapping.
        """
        return {}
