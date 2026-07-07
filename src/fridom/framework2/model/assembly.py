"""
The assembly pipeline and its static artifacts.

Description
-----------
Owning class spec: ``notes/framework2/model/classes/model.md``
(sections "ParameterBindingTable (+ Params)" and
"RematerializationTable"). `ParameterBindingTable` is the
assembly-frozen resolution of provides/requires (step 2): binding
rows map a dotted name to a live-leaf accessor ``(slot, attr)`` on a
provider — the time stepper joins as just another provider row, of
``fr.params.TIME_STEP`` (its ``dt`` leaf). Consumers never hold
provider objects or frozen values (the jax aliasing rule); `Params`
is the thin frozen in-trace mapping ``eval_params`` delivers per
stage time. `RematerializationTable` retains the AUXILIARY
declaration defaults in the static assembly record (the D1.1
softening): ``materialize`` is THE one code path shared by
allocation (step 8, all owners) and ``update_parameters`` (changed
owners only, host-writable entries skipped — CS-2).
"""
# Wave 3 A: ParameterBinding(Table), Params, RematerializationTable
from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.model.declarations import (
    Lifecycle,
    _leads_with_self,
)
from fridom.framework2.model.errors import (
    AssemblyError,
    MissingParameterError,
    ParameterCollisionError,
)
from fridom.framework2.model.parameters import (
    REQUIRED,
    USE_PROVIDED,
    Param,
    ParameterDeclaration,
    ParameterReference,
)
from fridom.framework2.model.params import TIME_STEP, ParamName
from fridom.framework2.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    import jax

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import (
        TensorProductSpace,
    )
    from fridom.framework2.model.declarations import FieldDeclaration

# used as "attribute absent" marker (None is a legal leaf value)
_MISSING = object()


def _module_label(slot: int, module: object) -> str:
    """Attribution label of one module slot."""
    return f"modules[{slot}] ({type(module).__name__})"


def _stepper_label(stepper: object) -> str:
    """Attribution label of the time stepper."""
    return f"the time stepper ({type(stepper).__name__})"


# ================================================================
#  ParameterBinding (one resolved row)
# ================================================================
class ParameterBinding(NamedTuple):

    """
    One resolved binding row: dotted name -> live-leaf accessor.

    Description
    -----------
    ``slot`` is the provider: a module tuple index, the string
    ``"stepper"`` (the time stepper — just another provider row), or
    ``None`` for an identity-defaulted constant entry (spec
    concretization: constant entries carry their value in ``value``
    and are listed in the report as "identity defaults in effect").

    Parameters
    ----------
    name : str
        The canonical dotted name (`ParamName` welcome).
    slot : int | Literal["stepper"] | None
        The provider slot; None marks a constant entry.
    attr : str
        The dynamic-leaf attribute on the provider ("" for
        constants).
    declaration : ParameterDeclaration
        Units/doc (the report; ``ParameterView.info``).
    value : object, optional
        The constant of an identity-defaulted entry (default: None).
    """

    name: str
    slot: int | Literal["stepper"] | None
    attr: str
    declaration: ParameterDeclaration
    value: object = None


# ================================================================
#  Params (the thin frozen in-trace mapping)
# ================================================================
@partial(jaxify, dynamic=("_values",))
class Params(Mapping):

    """
    Thin frozen mapping delivered to every traced entry point.

    Description
    -----------
    The product of ``eval_params``: fresh live leaves per stage time
    (`TimeDependent` values already resolved through
    ``fr.resolve_at``). Names live in the static treedef, values are
    dynamic leaves — zero-recompile parameter sweeps by
    construction. Unknown-name lookups are caught by the assembly
    dry run (the raise carries the bound-name list).

    Parameters
    ----------
    values : Mapping[str, object]
        The resolved ``name -> leaf`` mapping, binding order.
    """

    def __init__(self, values: Mapping[str, object]) -> None:
        """Freeze the names (static) and values (dynamic leaves)."""
        self._names: tuple[str, ...] = tuple(
            str(name) for name in values)
        self._values: tuple[object, ...] = tuple(values.values())

    def __getitem__(self, name: str) -> object:
        """
        Return the resolved leaf value of one bound name.

        Parameters
        ----------
        name : str
            The dotted parameter name.

        Returns
        -------
        object
            The leaf value (already stage-time resolved).

        Raises
        ------
        MissingParameterError
            If the name is not bound (caught by the dry run).
        """
        try:
            index = self._names.index(str(name))
        except ValueError:
            bound = ", ".join(self._names) or "none"
            raise MissingParameterError(
                f"parameter {str(name)!r} is not bound; bound "
                f"parameters: {bound}") from None
        return self._values[index]

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is bound."""
        return str(name) in self._names

    def __iter__(self) -> Iterator[str]:
        """Iterate the bound names, binding order."""
        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of bound names."""
        return len(self._names)

    def __repr__(self) -> str:
        """Summary repr listing the bound names."""
        return f"Params({', '.join(self._names)})"


# ================================================================
#  The host-side read-only mapping surface
# ================================================================
class _HostView:

    """
    Table-level read-only host mapping over live provider leaves.

    Description
    -----------
    Live leaf reads from the given providers; Ramp-valued slots
    return the `TimeDependent` object itself, never a
    silently-evaluated value (D2.4 — evaluate via ``at_time``). The
    full ``ParameterView`` (``at_time``/``info``) wraps this at the
    Model level (wave 4).

    Parameters
    ----------
    table : ParameterBindingTable
        The frozen binding table.
    modules : tuple
        The live module tuple (the carry's).
    stepper : object
        The live time stepper.
    """

    __slots__ = ("_modules", "_stepper", "_table")

    def __init__(
        self,
        table: ParameterBindingTable,
        modules: tuple,
        stepper: object,
    ) -> None:
        """Bind the view to live providers."""
        self._table = table
        self._modules = modules
        self._stepper = stepper

    def __getitem__(self, name: str) -> object:
        """
        Live leaf read of one bound name (Ramps returned raw).

        Parameters
        ----------
        name : str
            The dotted parameter name.

        Returns
        -------
        object
            The live leaf value; `TimeDependent` objects unevaluated.

        Raises
        ------
        MissingParameterError
            Hinted, on unprovided names.
        """
        entry = self._table[name]
        return _read_leaf(entry, self._modules, self._stepper)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is bound."""
        return name in self._table

    def __iter__(self) -> Iterator[str]:
        """Iterate the bound names, binding order."""
        return iter(self._table.names)

    def __len__(self) -> int:
        """Return the number of bound names."""
        return len(self._table)

    def __repr__(self) -> str:
        """Summary repr listing the bound names."""
        return f"<parameter view: {', '.join(self._table.names)}>"


def _read_leaf(
    entry: ParameterBinding,
    modules: tuple,
    stepper: object,
) -> object:
    """Read one binding's live leaf (raw, not time-resolved)."""
    if entry.slot is None:
        return entry.value
    provider = (stepper if entry.slot == "stepper"
                else modules[entry.slot])
    return getattr(provider, entry.attr)


# ================================================================
#  ParameterBindingTable (assembly step 2, frozen)
# ================================================================
class ParameterBindingTable:

    """
    Assembly-frozen resolution of provides/requires.

    Description
    -----------
    Built at assembly step 2 by :meth:`build`; frozen and hashable
    (part of the static ``AssemblyRecord``). The checks discharged
    at build: one provider per name (`ParameterCollisionError`
    naming both), unsatisfied `REQUIRED` references
    (`MissingParameterError` with the registry hint and the provided
    list), the no-default policy (a reference default on a
    ``no_default`` registry name is an error), provided leaves must
    be dynamic (jaxified providers), the duplicate-module aliasing
    lint, and the explicit-wins/`USE_PROVIDED` slot conversions.
    Identity-defaulted references bind to constant entries.

    Parameters
    ----------
    entries : Iterable[ParameterBinding]
        The resolved rows, binding order.
    """

    __slots__ = ("_by_name", "_entries")

    def __init__(
        self, entries: Iterable[ParameterBinding],
    ) -> None:
        """Validate one-provider-per-name and freeze."""
        entries = tuple(entries)
        by_name: dict[str, ParameterBinding] = {}
        for entry in entries:
            if not isinstance(entry, ParameterBinding):
                raise TypeError(
                    f"ParameterBindingTable rows are "
                    f"ParameterBindings, got {entry!r}")
            name = str(entry.name)
            other = by_name.get(name)
            if other is not None:
                raise ParameterCollisionError(
                    f"parameter {name!r} has two providers: slot "
                    f"{other.slot!r} and slot {entry.slot!r}; one "
                    "provider per name")
            by_name[name] = entry
        object.__setattr__(self, "_entries", entries)
        object.__setattr__(self, "_by_name", by_name)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation: the table is frozen after step 2."""
        raise AttributeError(
            "ParameterBindingTable is frozen after construction; "
            "build a new table instead")

    # ================================================================
    #  The step-2 resolution (build)
    # ================================================================
    @classmethod
    def build(
        cls,
        modules: tuple,
        stepper: object,
    ) -> ParameterBindingTable:
        """
        Resolve provides/requires over a module tuple + stepper.

        Description
        -----------
        Provider rows come from each module's
        ``parameter_declarations`` and from the stepper (which joins
        as provider of ``fr.params.TIME_STEP`` — its ``dt`` leaf —
        unless it publishes its own declarations). A declared slot
        whose live value is `USE_PROVIDED` converts to a `REQUIRED`
        reference; a slot holding an untouched `Param` default
        converts to a defaulted reference (explicit-wins — an
        explicit value is an owned value, no linkage claimed).
        References resolve against the provider rows; identity
        defaults bind to constant entries.

        Parameters
        ----------
        modules : tuple
            The module tuple (duck-typed: ``parameter_declarations``
            / ``parameter_references`` read when present).
        stepper : object
            The time stepper (duck-typed: its ``dt`` leaf).

        Returns
        -------
        ParameterBindingTable
            The frozen table.

        Raises
        ------
        ParameterCollisionError
            Two providers of one dotted name (names both).
        MissingParameterError
            An unsatisfied `REQUIRED` reference (registry hint +
            provided list).
        AssemblyError
            Aliased module objects, non-dynamic provided leaves, a
            default on a ``no_default`` name, or an unresolvable
            `USE_PROVIDED` slot.
        """
        _lint_aliased_modules(modules)
        rows: dict[str, ParameterBinding] = {}
        providers: dict[str, str] = {}
        references: list[tuple[str, ParameterReference]] = []
        for slot, module in enumerate(modules):
            _collect_module(module, slot, rows, providers,
                            references)
        _collect_stepper(stepper, rows, providers, references)
        for label, reference in references:
            _resolve_reference(label, reference, rows, providers)
        return cls(tuple(rows.values()))

    # ================================================================
    #  Mapping-style access
    # ================================================================
    @property
    def names(self) -> tuple[str, ...]:
        """The bound dotted names, binding order."""
        return tuple(str(entry.name) for entry in self._entries)

    def __getitem__(self, name: str) -> ParameterBinding:
        """
        Return the binding row of one dotted name.

        Parameters
        ----------
        name : str
            The dotted parameter name (`ParamName` welcome).

        Returns
        -------
        ParameterBinding
            The resolved row.

        Raises
        ------
        MissingParameterError
            With the registry hint and the provided list.
        """
        entry = self._by_name.get(str(name))
        if entry is None:
            hint = (f" (hint: {name.hint})"
                    if isinstance(name, ParamName) and name.hint
                    else "")
            provided = ", ".join(self.names) or "none"
            raise MissingParameterError(
                f"parameter {str(name)!r} is not provided{hint}; "
                f"provided parameters: {provided}")
        return entry

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is bound."""
        return str(name) in self._by_name

    def __iter__(self) -> Iterator[ParameterBinding]:
        """Iterate the binding rows, binding order."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Return the number of binding rows."""
        return len(self._entries)

    # ================================================================
    #  The reads (in-trace and host)
    # ================================================================
    def eval_params(
        self,
        modules: tuple,
        stepper: object,
        t: jax.Array | float,
    ) -> Params:
        """
        Resolve fresh live leaves at stage time (the in-trace read).

        Description
        -----------
        Composed once at assembly, called at P0 of every substage:
        every leaf is read from its live provider and resolved
        through ``fr.resolve_at`` at ``t`` (a ramped scalar and a
        ramped N² profile see the same time). Feeds
        ``StepContext.params``.

        Parameters
        ----------
        modules : tuple
            The live module tuple (the carry's).
        stepper : object
            The live time stepper.
        t : jax.Array | float
            The (possibly traced) stage time.

        Returns
        -------
        Params
            The frozen in-trace mapping.
        """
        return Params({
            str(entry.name): resolve_at(
                _read_leaf(entry, modules, stepper), t)
            for entry in self._entries})

    def host_view(
        self,
        modules: tuple,
        stepper: object,
    ) -> _HostView:
        """
        Return the read-only host mapping over live leaves.

        Description
        -----------
        Table-level: Ramp-valued slots return the Ramp object,
        never a silently-evaluated value; unknown names raise the
        hinted `MissingParameterError`. The full ``ParameterView``
        (``at_time``/``info``) wraps this at the Model level
        (wave 4).

        Parameters
        ----------
        modules : tuple
            The live module tuple.
        stepper : object
            The live time stepper.

        Returns
        -------
        _HostView
            The read-only mapping.
        """
        return _HostView(self, modules, stepper)

    # ================================================================
    #  Hashability (the table joins the static AssemblyRecord)
    # ================================================================
    def fingerprint_token(self) -> tuple:
        """
        Return the table's restart-fingerprint contribution.

        Description
        -----------
        Structure, never leaves: name, provider slot, attribute,
        and — for constant entries — the value's *spec* (its type
        name: a scalar/Ramp swap is structure, 02_rules).

        Returns
        -------
        tuple
            Human-diffable token rows.
        """
        return tuple(
            (str(entry.name), str(entry.slot), entry.attr,
             type(entry.value).__name__ if entry.slot is None
             else "")
            for entry in self._entries)

    def __eq__(self, other: object) -> bool:
        """Structural equality over the binding rows."""
        if not isinstance(other, ParameterBindingTable):
            return NotImplemented
        return self._entries == other._entries

    def __hash__(self) -> int:
        """Structural hash over the fingerprint token."""
        return hash(self.fingerprint_token())

    def __repr__(self) -> str:
        """Summary repr listing the bound names."""
        return (f"ParameterBindingTable("
                f"{', '.join(self.names) or 'empty'})")


# ================================================================
#  build() helpers (assembly step 2 mechanics)
# ================================================================
def _lint_aliased_modules(modules: tuple) -> None:
    """Reject aliased module objects (pytrees are trees, not DAGs)."""
    for i, module in enumerate(modules):
        for j in range(i + 1, len(modules)):
            if modules[j] is module:
                raise AssemblyError(
                    f"modules[{i}] and modules[{j}] are the same "
                    f"object ({type(module).__name__}); a module "
                    "appearing twice in the carry aliases its "
                    "leaves (pytrees are trees, not DAGs) — "
                    "construct two instances")


def _collect_module(
    module: object,
    slot: int,
    rows: dict[str, ParameterBinding],
    providers: dict[str, str],
    references: list[tuple[str, ParameterReference]],
) -> None:
    """Collect one module's provider rows and references."""
    label = _module_label(slot, module)
    declared_attrs: set[str] = set()
    for declaration in getattr(module, "parameter_declarations",
                               ()):
        declared_attrs.add(declaration.attr)
        _collect_provider(declaration, module, slot, label,
                          rows, providers, references)
    _scan_slots(module, label, declared_attrs, references)
    references.extend(
        (label, reference)
        for reference in getattr(module, "parameter_references",
                                 ()))


def _collect_stepper(
    stepper: object,
    rows: dict[str, ParameterBinding],
    providers: dict[str, str],
    references: list[tuple[str, ParameterReference]],
) -> None:
    """Join the stepper as provider of ``fr.params.TIME_STEP``."""
    label = _stepper_label(stepper)
    declarations = getattr(stepper, "parameter_declarations", None)
    if declarations is None:
        declarations = (ParameterDeclaration(
            TIME_STEP, attr="dt", units="s",
            doc="the stepper's time-step leaf"),)
    for declaration in declarations:
        _collect_provider(declaration, stepper, "stepper", label,
                          rows, providers, references)


def _collect_provider(
    declaration: ParameterDeclaration,
    provider: object,
    slot: int | Literal["stepper"],
    label: str,
    rows: dict[str, ParameterBinding],
    providers: dict[str, str],
    references: list[tuple[str, ParameterReference]],
) -> None:
    """Turn one declaration into a provider row (or a reference)."""
    name = str(declaration.name)
    if not declaration.attr:
        raise AssemblyError(
            f"{label} declares parameter {name!r} without attr=; a "
            "declaration names where the live value lives")
    value = getattr(provider, declaration.attr, _MISSING)
    if value is _MISSING:
        raise AssemblyError(
            f"{label} declares parameter {name!r} on attribute "
            f"{declaration.attr!r}, which it does not have")
    if value is USE_PROVIDED:
        # the converse sentinel: the slot resolves through the table
        references.append((label, ParameterReference(
            declaration.name, default=REQUIRED)))
        return
    if isinstance(value, Param):
        # untouched Param default: a defaulted reference, no provide
        references.append((label, ParameterReference(
            value.name, default=value.default)))
        return
    dynamic = getattr(provider, "dynamic_jax_attrs", None)
    if dynamic is not None and declaration.attr not in dynamic:
        raise AssemblyError(
            f"{label} provides parameter {name!r} from the static "
            f"attribute {declaration.attr!r}; provided parameters "
            "must be dynamic leaves (a static leaf silently "
            "recompiles per sweep)")
    other = providers.get(name)
    if other is not None:
        raise ParameterCollisionError(
            f"parameter {name!r} has two providers: {other} and "
            f"{label}; one provider per name")
    providers[name] = label
    rows[name] = ParameterBinding(
        name=declaration.name, slot=slot, attr=declaration.attr,
        declaration=declaration)


def _scan_slots(
    module: object,
    label: str,
    declared_attrs: set[str],
    references: list[tuple[str, ParameterReference]],
) -> None:
    """Find `Param`/`USE_PROVIDED` constructor slots (consumers)."""
    for attr, value in getattr(module, "__dict__", {}).items():
        if attr in declared_attrs:
            continue  # handled by _collect_provider
        if isinstance(value, Param):
            references.append((label, ParameterReference(
                value.name, default=value.default)))
        elif value is USE_PROVIDED:
            raise AssemblyError(
                f"{label} passes USE_PROVIDED in slot {attr!r}, "
                "which no ParameterDeclaration names; the sentinel "
                "forces a normally-owned (declared) slot through "
                "the binding table")


def _resolve_reference(
    label: str,
    reference: ParameterReference,
    rows: dict[str, ParameterBinding],
    providers: dict[str, str],
) -> None:
    """Resolve one reference against the provider rows."""
    name = str(reference.name)
    if name in providers:
        return
    if reference.default is REQUIRED:
        hint = reference.hint or (
            reference.name.hint
            if isinstance(reference.name, ParamName) else "")
        hint = f" (hint: {hint})" if hint else ""
        provided = ", ".join(sorted(providers)) or "none"
        raise MissingParameterError(
            f"{label} requires the parameter {name!r}, which "
            f"nothing provides{hint}; provided parameters: "
            f"{provided}")
    if (isinstance(reference.name, ParamName)
            and reference.name.no_default):
        raise AssemblyError(
            f"{label} declares a default for {name!r}, a registry "
            "name marked no_default (a default here would silently "
            "change the physics); provide it or drop the default")
    existing = rows.get(name)
    if existing is not None:
        # a previous identity-defaulted reference bound a constant
        if existing.value is not reference.default \
                and existing.value != reference.default:
            raise AssemblyError(
                f"conflicting identity defaults for {name!r}: "
                f"{existing.value!r} and {reference.default!r} "
                f"(the second from {label}); defaults on one name "
                "must agree")
        return
    rows[name] = ParameterBinding(
        name=reference.name, slot=None, attr="",
        declaration=ParameterDeclaration(
            reference.name,
            doc="identity default at the reference site"),
        value=reference.default)


# ================================================================
#  RematerializationTable (assembly step 5, frozen)
# ================================================================
@dataclass(frozen=True)
class RematerializationEntry:

    """
    One retained AUXILIARY declaration default.

    Description
    -----------
    The D1.1 softening: declarations are transient, but AUXILIARY
    ``default=`` values are retained in the static assembly record
    because ``update_parameters`` re-runs them. The default may be
    an UNBOUND owner method ``(self, grid, space)``; bound methods
    are rejected (the D2 aliasing lint — a bound method captures the
    assembly-time instance while live parameters ride the carry).

    Parameters
    ----------
    field : str
        The AUXILIARY component name.
    owner : int
        The owning module's tuple index.
    default : Callable | float | None
        The RETAINED declaration default (any of the four forms).
    space : TensorProductSpace
        The resolved bare space.
    host_writable : bool, optional
        Exempt from re-runs (CS-2): the host write is the source of
        truth; the default is initialization-only (default: False).
    """

    field: str
    owner: int
    default: Callable | float | None
    space: TensorProductSpace
    host_writable: bool = False

    def __post_init__(self) -> None:
        """Reject bound-method defaults (the aliasing lint)."""
        if (callable(self.default)
                and getattr(self.default, "__self__", None)
                is not None):
            raise TypeError(
                f"field {self.field!r}: default= callables are "
                "stored UNBOUND and paired with the owner slot; "
                f"{self.default!r} is bound and would capture the "
                "assembly-time instance while live parameters ride "
                "the carry (the D2 aliasing trap). Pass the class "
                "attribute instead")

    @classmethod
    def from_declaration(
        cls,
        declaration: FieldDeclaration,
        *,
        owner: int,
        space: TensorProductSpace,
    ) -> RematerializationEntry:
        """
        Retain one AUXILIARY declaration's default.

        Parameters
        ----------
        declaration : FieldDeclaration
            An AUXILIARY declaration.
        owner : int
            The owning module's tuple index.
        space : TensorProductSpace
            The resolved bare space (from the field table).

        Returns
        -------
        RematerializationEntry
            The retained row.

        Raises
        ------
        ValueError
            If the declaration is not AUXILIARY (only AUX defaults
            are retained; DIAGNOSTIC defaults are ``reset()``'s).
        """
        if declaration.lifecycle is not Lifecycle.AUXILIARY:
            raise ValueError(
                f"field {declaration.name!r} is "
                f"{declaration.lifecycle.name}; the "
                "re-materialization table retains AUXILIARY "
                "declaration defaults only")
        return cls(
            field=declaration.name, owner=owner,
            default=declaration.default, space=space,
            host_writable=declaration.host_writable)


@dataclass(frozen=True)
class RematerializationTable:

    """
    AUXILIARY declaration defaults, retained in the static record.

    Description
    -----------
    :meth:`materialize` is THE one code path shared by allocation
    (assembly step 8, all owners) and ``update_parameters`` (changed
    owners only) — that identity is what makes re-materialization
    correct by construction (02_rules, defaults-one-path).
    Invalidation is per-owner and conservative; host-writable
    entries are carried but flagged exempt (CS-2). Ramp-fed AUX
    allocation values are placeholders — ``self_update`` owns them
    in-run.

    Parameters
    ----------
    entries : tuple[RematerializationEntry, ...], optional
        The retained rows, declaration order (default: ()).
    """

    entries: tuple[RematerializationEntry, ...] = ()

    def __post_init__(self) -> None:
        """Validate row types and per-field uniqueness."""
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, RematerializationEntry):
                raise TypeError(
                    f"RematerializationTable rows are "
                    f"RematerializationEntries, got {entry!r}")
            if entry.field in seen:
                raise AssemblyError(
                    f"field {entry.field!r} appears twice in the "
                    "re-materialization table")
            seen.add(entry.field)

    def materialize(
        self,
        modules: tuple,
        grid: Grid,
        *,
        owners: frozenset[int] | set[int] | None = None,
    ) -> dict[str, ScalarField]:
        """
        Run the owners' defaults with their CURRENT leaves.

        Description
        -----------
        The ONE shared code path: ``owners=None`` is the allocation
        call (step 8 — every entry, including host-writable ones,
        whose defaults are initialization-only); an owner set is the
        ``update_parameters`` call (only entries of changed owners,
        host-writable entries skipped — CS-2). A default closure
        reads only the owner's own leaves (02_rules;
        cross-module-derived AUX values are ``self_update``
        territory or a re-assembly).

        Parameters
        ----------
        modules : tuple
            The live module tuple (the carry's — current leaves).
        grid : Grid
            The grid the fields materialize on.
        owners : frozenset[int] | set[int] | None, optional
            Changed owner slots (``update_parameters``), or None
            for the allocation pass (default: None).

        Returns
        -------
        dict[str, ScalarField]
            The materialized fields, entry order.
        """
        fields: dict[str, ScalarField] = {}
        for entry in self.entries:
            if owners is not None and (
                    entry.owner not in owners
                    or entry.host_writable):
                continue
            fields[entry.field] = _materialize_entry(
                entry, modules[entry.owner], grid)
        return fields

    def fingerprint_token(self) -> tuple:
        """
        Return the restart-fingerprint contribution (structure).

        Returns
        -------
        tuple
            ``(field, owner, default form, space)`` token rows.
        """
        return tuple(
            (entry.field, entry.owner,
             _default_form(entry.default), repr(entry.space),
             entry.host_writable)
            for entry in self.entries)


def _default_form(default: Callable | float | None) -> str:
    """Disambiguate the default form (mirrors declarations)."""
    if default is None:
        return "zeros"
    if not callable(default):
        return "constant"
    if _leads_with_self(default):
        return "owner_method"
    return "coordinate"


def _materialize_entry(
    entry: RematerializationEntry,
    module: object,
    grid: Grid,
) -> ScalarField:
    """
    Evaluate one entry's default — the one shared path.

    Description
    -----------
    The four ``default=`` forms of D1.1: None is zeros; a number is
    a constant fill; an unbound owner method ``(self, grid, space)``
    is called with the *live* module; any other callable is a
    coordinate function routed through ``grid.create_field``.

    Parameters
    ----------
    entry : RematerializationEntry
        The retained row.
    module : object
        The live owning module (current leaves).
    grid : Grid
        The grid to materialize on.

    Returns
    -------
    ScalarField
        The materialized field.
    """
    default = entry.default
    if default is None:
        return grid.create_field(entry.space, name=entry.field)
    if isinstance(default, numbers.Number):
        data = jnp.full(entry.space.shape, default)
        return grid.create_field(entry.space, data=data,
                                 name=entry.field)
    if _leads_with_self(default):
        field = default(module, grid, entry.space)
        if not isinstance(field, ScalarField):
            raise TypeError(
                f"field {entry.field!r}: the owner-method default "
                f"of {type(module).__name__} must return a "
                f"ScalarField, got {field!r}")
        return field
    return grid.create_field(entry.space, init=default,
                             name=entry.field)


# ================================================================
#  Wave 4: the nine-step pipeline function, AssemblyRecord, and
#  Fingerprint land here (model.md sections 2, "AssemblyRecord",
#  and "AssemblyReport (+ Fingerprint)").
# ================================================================
