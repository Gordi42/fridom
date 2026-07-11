"""
Field declarations of the model layer.

Description
-----------
Owning class spec: ``design/specs/model/classes/declarations.md``
("Lifecycle"; "FieldDeclaration"; "FieldReference").
``FieldDeclaration`` is a module's claim on one state component —
plain frozen host data (never a pytree, never in the carry),
consumed by assembly step 1 and discarded, with one signed
exception: AUXILIARY declarations with callable ``default=`` are
retained in the static re-materialization table (D4 amendment to
D1.1). ``Lifecycle`` is the closed structural axis (who advances the
field); ``FieldReference`` is a consumer's checked claim on a field
it does not own (D1.5). Validation homes: the local validity checks
(dot-free name, role/lifecycle compatibility, ``host_writable``
gating) run here at construction; collision/coverage checks execute
in assembly (model cluster).
"""
# Wave 2 A: Lifecycle, FieldDeclaration (+ tracer/velocity
#    templates), FieldReference
from __future__ import annotations

import dataclasses
import inspect
import numbers
import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple, final

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.model.roles import ADVECTED, TRACER, Role, Velocity
from fridom.framework2.model.space_patterns import (
    Collocated,
    SpacePattern,
    SpaceRule,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping


class Lifecycle(Enum):

    """
    Who advances this field; what stepper/restart/IO branch on.

    Description
    -----------
    The closed, mandatory lifecycle axis of every declaration
    (model D1.4; three members, final): PROGNOSTIC fields are
    advanced by the stepper from tendency contributions (valid at
    step start, in restart); AUXILIARY fields are module-owned carry
    data, written only by their owning module, never advected/mixed
    regardless of roles; DIAGNOSTIC fields are written during the
    step by a stage and read by IO or later stages (reads see the
    nearest preceding write in schedule order). Deliberately not a
    role: it is the one tag every stepper must interpret.
    """

    PROGNOSTIC = auto()
    AUXILIARY = auto()
    DIAGNOSTIC = auto()


class FieldDeclaration:

    """
    Frozen declaration of one field: a module's state-vector claim.

    Description
    -----------
    Name, space pattern, lifecycle, roles, background default,
    consent flags, annotation (model D1.1). A plain frozen host
    object — the collision unit (exactly one owner per name) and the
    metadata source; never a pytree. ``default=`` is the background
    initializer, not the IC mechanism: None means zeros, a number a
    constant fill, a coordinate callable ``f(x, y, ...)`` is routed
    through ``grid.create_field``, and an unbound owner method
    ``(self, grid, space)`` is called with the live module (the D4
    amendment) — the forms are disambiguated by inspection
    (:attr:`default_form`). Every callable is stored UNBOUND: bound
    methods are rejected (the D2 aliasing trap).

    Parameters
    ----------
    name : str
        The flat, dot-free field name (dots are the parameter
        namespace separator, D2.1).
    space : SpacePattern | SpaceRule
        The grid-free space descriptor (keyword-only, mandatory).
    lifecycle : Lifecycle, optional
        Who advances the field (default: Lifecycle.PROGNOSTIC).
    roles : Iterable[Role], optional
        Opt-in consumer tags; normalized to a frozenset
        (default: ()).
    default : float | Callable | None, optional
        The background initializer (default: None, zeros).
    host_writable : bool, optional
        The owner's consent for host-side chunk-boundary
        ``set_aux`` writes; AUXILIARY/DIAGNOSTIC only (CS-1)
        (default: False).
    long_name : str, optional
        Descriptive nc-style name (default: "Unnamed").
    units : str, optional
        Physical units annotation (default: "n/a").
    nc_attrs : Mapping[str, str] | None, optional
        Extra netCDF attributes; normalized to sorted tuple pairs
        (default: None).
    """

    __slots__ = ("_default", "_host_writable", "_lifecycle",
                 "_long_name", "_name", "_nc_attrs", "_roles",
                 "_space", "_units")

    def __init__(
        self,
        name: str,
        *,
        space: SpacePattern | SpaceRule,
        lifecycle: Lifecycle = Lifecycle.PROGNOSTIC,
        roles: Iterable[Role] = (),
        default: float | Callable | None = None,
        host_writable: bool = False,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> None:
        """Normalize and locally validate the declaration."""
        self._name: str = _check_name(name)
        self._space: SpacePattern | SpaceRule = _check_space(
            name, space)
        if not isinstance(lifecycle, Lifecycle):
            raise TypeError(
                f"field {name!r}: lifecycle must be a Lifecycle "
                f"member, got {lifecycle!r}")
        self._lifecycle: Lifecycle = lifecycle
        self._roles: frozenset[Role] = _check_roles(
            name, lifecycle, roles)
        self._default: float | Callable | None = _check_default(
            name, default)
        if not isinstance(host_writable, bool):
            raise TypeError(
                f"field {name!r}: host_writable must be a bool, "
                f"got {host_writable!r}")
        if host_writable and lifecycle is Lifecycle.PROGNOSTIC:
            raise ValueError(
                f"field {name!r}: host_writable consent is "
                "lifecycle-polymorphic over AUXILIARY and "
                "DIAGNOSTIC only (CS-1); a PROGNOSTIC field is "
                "advanced by the stepper, never host-written")
        self._host_writable: bool = host_writable
        self._long_name: str = _check_str(name, "long_name",
                                          long_name)
        self._units: str = _check_str(name, "units", units)
        self._nc_attrs: tuple[tuple[str, str], ...] = (
            _normalize_nc_attrs(nc_attrs))

    # ================================================================
    #  Read-only attributes
    # ================================================================
    @property
    def name(self) -> str:
        """The flat, dot-free field name."""
        return self._name

    @property
    def space(self) -> SpacePattern | SpaceRule:
        """The grid-free space descriptor (pattern or rule)."""
        return self._space

    @property
    def lifecycle(self) -> Lifecycle:
        """Who advances the field."""
        return self._lifecycle

    @property
    def roles(self) -> frozenset[Role]:
        """The opt-in consumer tags (normalized frozenset)."""
        return self._roles

    @property
    def default(self) -> float | Callable | None:
        """The background initializer (not the IC mechanism)."""
        return self._default

    @property
    def host_writable(self) -> bool:
        """Owner consent for host-side ``set_aux`` writes (CS-1)."""
        return self._host_writable

    @property
    def long_name(self) -> str:
        """Descriptive nc-style name."""
        return self._long_name

    @property
    def units(self) -> str:
        """Physical units annotation (documentation only)."""
        return self._units

    @property
    def nc_attrs(self) -> tuple[tuple[str, str], ...]:
        """Extra netCDF attributes as sorted (key, value) pairs."""
        return self._nc_attrs

    @property
    def default_form(self) -> str:
        """
        The disambiguated ``default=`` form.

        Description
        -----------
        One of ``"zeros"`` (None), ``"constant"`` (a number),
        ``"coordinate"`` (a callable matched against coordinate
        names), or ``"owner_method"`` (an unbound owner method).
        The two callable forms are told apart by inspection: a first
        positional parameter named ``self`` selects the owner-method
        form (spec concretization, confirmed at 2.2).
        """
        if self._default is None:
            return "zeros"
        if not callable(self._default):
            return "constant"
        if _leads_with_self(self._default):
            return "owner_method"
        return "coordinate"

    # ================================================================
    #  Templates (the registration acts of D1.4)
    # ================================================================
    @classmethod
    def tracer(
        cls,
        name: str,
        *,
        space: SpacePattern | SpaceRule | None = None,
        default: float | Callable | None = None,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> FieldDeclaration:
        """
        Template: PROGNOSTIC + {TRACER, ADVECTED}.

        Description
        -----------
        The registration act of D1.4: a declared tracer is advected
        by the advection scheme(s) and targeted by TRACER-selecting
        closures. ``space`` defaults to ``Collocated()``.

        Parameters
        ----------
        name : str
            The flat, dot-free field name.
        space : SpacePattern | SpaceRule | None, optional
            The space descriptor (default: None -> Collocated()).
        default : float | Callable | None, optional
            The background initializer (default: None).
        long_name : str, optional
            Descriptive nc-style name (default: "Unnamed").
        units : str, optional
            Physical units annotation (default: "n/a").
        nc_attrs : Mapping[str, str] | None, optional
            Extra netCDF attributes (default: None).

        Returns
        -------
        FieldDeclaration
            The tracer declaration.
        """
        return cls(
            name,
            space=Collocated() if space is None else space,
            lifecycle=Lifecycle.PROGNOSTIC,
            roles=(TRACER, ADVECTED),
            default=default, long_name=long_name, units=units,
            nc_attrs=nc_attrs)

    @classmethod
    def velocity(
        cls,
        name: str,
        component: str,
        *,
        space: SpacePattern | SpaceRule,
        default: float | Callable | None = None,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> FieldDeclaration:
        """
        Template: PROGNOSTIC + {Velocity(component), ADVECTED}.

        Description
        -----------
        Derives the universal wall condition (C8, topology-driven
        walls): the wall-normal velocity is Dirichlet on its own
        bounded component axis (impermeability), so a ``wall_bc``
        Dirichlet entry for ``component`` is injected into a
        ``SpacePattern`` space. The entry is topology-conditional —
        on periodic grids it is inert and the pattern resolves to
        the identical interned spaces. A pattern that already pins
        the component axis (in ``bc`` or ``wall_bc``) is used as
        given, and a ``SpaceRule`` passes through untouched (the
        full-power escape hatch stays the declarer's
        responsibility).

        Parameters
        ----------
        name : str
            The flat, dot-free field name.
        component : str
            The explicit velocity component label (never
            space-derived).
        space : SpacePattern | SpaceRule
            The space descriptor (mandatory: staggering is the
            declarer's choice).
        default : float | Callable | None, optional
            The background initializer (default: None).
        long_name : str, optional
            Descriptive nc-style name (default: "Unnamed").
        units : str, optional
            Physical units annotation (default: "n/a").
        nc_attrs : Mapping[str, str] | None, optional
            Extra netCDF attributes (default: None).

        Returns
        -------
        FieldDeclaration
            The velocity-component declaration.
        """
        if isinstance(space, SpacePattern):
            space = _with_wall_dirichlet(space, component)
        return cls(
            name, space=space, lifecycle=Lifecycle.PROGNOSTIC,
            roles=(Velocity(component), ADVECTED),
            default=default, long_name=long_name, units=units,
            nc_attrs=nc_attrs)

    # ================================================================
    #  Functional update / metadata folding
    # ================================================================
    def replace(self, **changes: object) -> FieldDeclaration:
        """
        Functional update (module factories, preset tweaks).

        Parameters
        ----------
        **changes : object
            Constructor keywords to replace; ``name`` may be given
            positionally there, everything is re-validated.

        Returns
        -------
        FieldDeclaration
            The updated declaration; ``self`` is unchanged.
        """
        kwargs: dict[str, object] = {
            "name": self._name,
            "space": self._space,
            "lifecycle": self._lifecycle,
            "roles": self._roles,
            "default": self._default,
            "host_writable": self._host_writable,
            "long_name": self._long_name,
            "units": self._units,
            "nc_attrs": self._nc_attrs,
        }
        for key, value in changes.items():
            if key not in kwargs:
                raise TypeError(
                    f"replace() got an unexpected keyword {key!r}; "
                    f"valid keys: {tuple(kwargs)}")
            kwargs[key] = value
        name = kwargs.pop("name")
        return type(self)(name, **kwargs)  # type: ignore[arg-type]

    def field_metadata(self) -> FieldMetadata:
        """
        Fold the annotation into the grid-layer metadata record.

        Description
        -----------
        What survives allocation: name/long_name/units/nc_attrs as
        the ``FieldMetadata`` attached to the allocated field. The
        declaration itself is transient assembly data.

        Returns
        -------
        FieldMetadata
            The annotation record.
        """
        return FieldMetadata(
            name=self._name, long_name=self._long_name,
            units=self._units, nc_attrs=self._nc_attrs)

    def __repr__(self) -> str:
        """Render the non-default declaration data."""
        parts = [repr(self._name), f"space={self._space!r}",
                 f"lifecycle=Lifecycle.{self._lifecycle.name}"]
        if self._roles:
            roles = ", ".join(sorted(
                repr(role) for role in self._roles))
            parts.append(f"roles=({roles})")
        if self._default is not None:
            parts.append(f"default={self._default!r}")
        if self._host_writable:
            parts.append("host_writable=True")
        return f"FieldDeclaration({', '.join(parts)})"


@final
class FieldReference(NamedTuple):

    """
    Checked at assembly; ``MissingFieldError`` carries the hint.

    Description
    -----------
    A consumer's declared claim on a field it does not own (D1.5):
    declared in ``Module.field_references``, checked at assembly
    step 1, attributed to the requiring module on failure. No
    auto-creation and deliberately no space slot — a reference
    carrying a space is a declaration in disguise; a module that
    requires a specific BC/space checks the resolved space at
    ``bind(table)``.

    Parameters
    ----------
    name : str
        The required field name.
    hint : str, optional
        Human help attached to the missing-field error, e.g.
        "velocities are declared by a dynamical-core module"
        (default: "").
    """

    name: str
    hint: str = ""


# ================================================================
#  Wall derivation (C8: topology-driven walls)
# ================================================================
def _with_wall_dirichlet(
    space: SpacePattern, component: str,
) -> SpacePattern:
    """
    Inject the impermeability wall BC for one component axis.

    Description
    -----------
    Adds ``wall_bc[component] = BC.DIRICHLET`` — applied at
    resolution only where the component's mesh factor is bounded —
    unless the pattern already pins that axis in ``bc`` or
    ``wall_bc`` (the declarer's explicit choice wins).

    Parameters
    ----------
    space : SpacePattern
        The declared velocity space pattern.
    component : str
        The velocity component (coordinate) name.

    Returns
    -------
    SpacePattern
        The pattern with the derived wall entry (``space`` itself
        when the axis is already pinned).
    """
    pinned = ({name for name, _ in space.bc}
              | {name for name, _ in space.wall_bc})
    if component in pinned:
        return space
    return dataclasses.replace(
        space, wall_bc=(*space.wall_bc, (component, BC.DIRICHLET)))


# ================================================================
#  Local validity checks (assembly re-checks collisions/coverage)
# ================================================================
def _check_name(name: str) -> str:
    """Validate the flat, dot-free field name."""
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"field names are non-empty strings, got {name!r}")
    if "." in name:
        raise ValueError(
            f"field name {name!r} contains a dot; field names are "
            "flat (prefix-by-convention, D1.5) — dots are the "
            "parameter namespace separator (D2.1)")
    return name


def _check_space(
    name: str, space: object,
) -> SpacePattern | SpaceRule:
    """Validate the space slot (pattern or rule; one protocol)."""
    if isinstance(space, SpacePattern | SpaceRule):
        return space
    raise TypeError(
        f"field {name!r}: space must be a SpacePattern (Collocated/"
        f"Staggered/Profile) or a SpaceRule, got {space!r}")


def _check_roles(
    name: str, lifecycle: Lifecycle, roles: Iterable[Role],
) -> frozenset[Role]:
    """Normalize roles; enforce role/lifecycle compatibility."""
    normalized = frozenset(roles)
    for role in normalized:
        if not isinstance(role, Role):
            raise TypeError(
                f"field {name!r}: roles must be Role instances, "
                f"got {role!r}")
    if lifecycle is Lifecycle.AUXILIARY and normalized:
        raise ValueError(
            f"field {name!r}: AUXILIARY fields carry no roles — "
            "they are module-owned carry data, never advected or "
            "mixed (D1.4)")
    if lifecycle is Lifecycle.DIAGNOSTIC:
        illegal = tuple(sorted(
            repr(role) for role in normalized
            if not isinstance(role, Velocity)))
        if illegal:
            raise ValueError(
                f"field {name!r}: DIAGNOSTIC fields admit only the "
                f"Velocity role (V-H2, the diagnosed w); got "
                f"{illegal}")
    if (lifecycle is Lifecycle.PROGNOSTIC and TRACER in normalized
            and ADVECTED not in normalized):
        warnings.warn(
            f"field {name!r} declares TRACER without ADVECTED: it "
            "will be mixed but not transported (suppress this "
            "warning if intentional)",
            stacklevel=3)
    return normalized


def _check_default(
    name: str, default: object,
) -> float | Callable | None:
    """
    Validate the background-initializer slot (accept-and-defer).

    Description
    -----------
    Callables are accepted here regardless of bound-ness: the
    owner-identity check is deferred to assembly, where the owning
    module is known
    (``RematerializationEntry.from_declaration(owner=...)``). A bound
    method of the OWNING module is the natural ``default=self._make``
    spelling and normalizes to its ``__func__``; a bound method of
    any OTHER object is the D2 aliasing trap, rejected at assembly.
    """
    if default is None or isinstance(default, numbers.Number):
        return default
    if callable(default):
        return default
    raise TypeError(
        f"field {name!r}: default= takes None (zeros), a number "
        f"(constant fill), or a callable, got {default!r}")


def _leads_with_self(fn: Callable) -> bool:
    """Whether the callable's first positional parameter is self."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    for parameter in parameters.values():
        return (parameter.kind in positional
                and parameter.name == "self")
    return False


def _check_str(name: str, label: str, value: str) -> str:
    """Validate a string annotation slot."""
    if not isinstance(value, str):
        raise TypeError(
            f"field {name!r}: {label} must be a string, got "
            f"{value!r}")
    return value


def _normalize_nc_attrs(
    nc_attrs: Mapping[str, str] | tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[str, str], ...]:
    """Normalize nc-attrs to the canonical sorted tuple of pairs."""
    if nc_attrs is None:
        return ()
    if isinstance(nc_attrs, tuple):
        return tuple(sorted(nc_attrs))
    return tuple(sorted(nc_attrs.items()))
