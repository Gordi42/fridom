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
# Wave 4 A: assemble() (the nine-step pipeline, steps 1-7 + 9),
#    AssemblyArtifacts, AssemblyRecord, Fingerprint
from __future__ import annotations

import hashlib
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.model.composer import TendencyComposer
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.declarations import (
    Lifecycle,
    _leads_with_self,
)
from fridom.framework2.model.errors import (
    AssemblyError,
    MissingParameterError,
    ParameterCollisionError,
)
from fridom.framework2.model.field_table import FieldRecord, FieldTable
from fridom.framework2.model.module import BindParameterView
from fridom.framework2.model.parameters import (
    REQUIRED,
    USE_PROVIDED,
    Param,
    ParameterDeclaration,
    ParameterReference,
)
from fridom.framework2.model.params import TIME_STEP, ParamName
from fridom.framework2.model.report import (
    RUN_START_PLACEHOLDER,
    AssemblyReport,
)
from fridom.framework2.model.roles import ADVECTED
from fridom.framework2.model.schedule import (
    Schedule,
    apply_replace,
    evaluate_entry,
)
from fridom.framework2.model.space_patterns import (
    SpacePattern,
    SpaceRule,
)
from fridom.framework2.model.stages import StageKind
from fridom.framework2.model.terms import TendencyTerm, Treatment
from fridom.framework2.model.time_dependent import (
    TimeDependent,
    resolve_at,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    import jax

    from fridom.framework2.grid.decomposition.decomposition import (
        ReshardingReport,
    )
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import (
        SpaceLike,
        TensorProductSpace,
    )
    from fridom.framework2.model.declarations import FieldDeclaration
    from fridom.framework2.model.stages import Stage

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
        owner_instance: object = None,
    ) -> RematerializationEntry:
        """
        Retain one AUXILIARY declaration's default.

        Description
        -----------
        The default may be a bound method of the owning module (the
        natural ``default=self._make`` spelling): given the owning
        instance, it is normalized to its ``__func__`` so the retained
        default stays UNBOUND and is called ``default(module, grid,
        space)`` with the *live* module at re-materialization. A bound
        method of any OTHER object is the D2 aliasing trap.

        Parameters
        ----------
        declaration : FieldDeclaration
            An AUXILIARY declaration.
        owner : int
            The owning module's tuple index.
        space : TensorProductSpace
            The resolved bare space (from the field table).
        owner_instance : object, optional
            The owning module instance; enables normalizing a bound
            owner-method default to unbound (default: None — no
            normalization, an unbound default is required).

        Returns
        -------
        RematerializationEntry
            The retained row.

        Raises
        ------
        ValueError
            If the declaration is not AUXILIARY (only AUX defaults
            are retained; DIAGNOSTIC defaults are ``reset()``'s).
        TypeError
            If the default is a bound method of a different object
            (the D2 aliasing trap).
        """
        if declaration.lifecycle is not Lifecycle.AUXILIARY:
            raise ValueError(
                f"field {declaration.name!r} is "
                f"{declaration.lifecycle.name}; the "
                "re-materialization table retains AUXILIARY "
                "declaration defaults only")
        default = _normalize_default(
            declaration.name, declaration.default, owner_instance)
        return cls(
            field=declaration.name, owner=owner,
            default=default, space=space,
            host_writable=declaration.host_writable)


def _normalize_default(
    field: str, default: object, owner_instance: object,
) -> object:
    """
    Accept a bound owner-method default; normalize it to unbound.

    Description
    -----------
    A bound method of the OWNING module (``default=self._make``) is
    normalized to its ``__func__`` so the retained default is stored
    UNBOUND and called ``default(module, grid, space)`` with the live
    module (the one shared re-materialization path). A bound method of
    any other object is the D2 aliasing trap — it would capture the
    assembly-time instance while live parameters ride the carry.

    Parameters
    ----------
    field : str
        The AUXILIARY component name (error attribution).
    default : object
        The declaration's retained default (any of the four forms).
    owner_instance : object
        The owning module instance, or None (no normalization).

    Returns
    -------
    object
        The unbound default (``default.__func__`` for a bound owner
        method, else ``default`` unchanged).

    Raises
    ------
    TypeError
        If ``default`` is a bound method of a different object.
    """
    if not (callable(default)
            and getattr(default, "__self__", None) is not None):
        return default
    if owner_instance is not None and default.__self__ is owner_instance:
        return default.__func__
    raise TypeError(
        f"field {field!r}: default= callables are stored UNBOUND "
        f"and paired with the owner slot; {default!r} is bound to a "
        "different object and would capture the assembly-time "
        "instance while live parameters ride the carry (the D2 "
        "aliasing trap). Pass the class attribute instead")


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
#  Wave 4: the nine-step assembly pipeline (steps 1-7 + 9; step 8,
#  carry allocation, is wave 4.2), AssemblyRecord, Fingerprint,
#  AssemblyArtifacts (model.md sections 2, "AssemblyRecord",
#  "AssemblyReport (+ Fingerprint)").
# ================================================================


# ================================================================
#  Small assembly-time views
# ================================================================
class _StateSpaceMapping:

    """
    Name-keyed state spaces handed to ``grid.negotiate``.

    Description
    -----------
    The halo trace needs a NAME-KEYED tracer state (attribution and
    the per-field negotiation key on component names — the Phase-2
    reconciliation amendment), so ``trace_halo`` consumes ``items()``.
    The grid's bookkeeping paths, however, iterate the state spaces
    directly (``for s in state_spaces``); this view therefore
    iterates its VALUES, satisfying both consumers without touching
    grid code.

    Parameters
    ----------
    spaces : Mapping[str, SpaceLike]
        The resolved bare spaces, keyed by declared field name.
    """

    __slots__ = ("_spaces",)

    def __init__(self, spaces: Mapping[str, SpaceLike]) -> None:
        """Freeze the name -> space mapping (declaration order)."""
        self._spaces: dict[str, SpaceLike] = dict(spaces)

    def items(self) -> Iterable[tuple[str, SpaceLike]]:
        """Return the (name, space) pairs (the halo-trace side)."""
        return self._spaces.items()

    def keys(self) -> Iterable[str]:
        """Return the declared field names."""
        return self._spaces.keys()

    def values(self) -> Iterable[SpaceLike]:
        """Return the resolved bare spaces."""
        return self._spaces.values()

    def __getitem__(self, name: str) -> SpaceLike:
        """Return the space of one declared field."""
        return self._spaces[name]

    def __iter__(self) -> Iterator[SpaceLike]:
        """Iterate the SPACES (grid bookkeeping iterates directly)."""
        return iter(self._spaces.values())

    def __len__(self) -> int:
        """Return the number of declared fields."""
        return len(self._spaces)

    def __repr__(self) -> str:
        """Compact name-keyed summary."""
        return f"_StateSpaceMapping({sorted(self._spaces)!r})"


class _BindTable:

    """
    The bind-time table handed to ``Module.bind`` (step 4).

    Description
    -----------
    The resolved `FieldTable` query surface plus the ``parameters``
    attribute: a `BindParameterView` over the binding table's live
    leaves — bare reads of `TimeDependent` values raise
    ``TimeDependentParameterError`` unless spelled ``at_time(0.0)``
    (the bind/in-step split, D2.1). Everything else delegates to the
    frozen field table, including ``grid`` (which exposes the
    registry AS MERGED — step 3 runs before step 4, load-bearing).

    Parameters
    ----------
    table : FieldTable
        The resolved, frozen field table (step 1).
    parameters : BindParameterView
        The gated bind-time parameter view (step 2 values).
    """

    __slots__ = ("_table", "parameters")

    def __init__(
        self,
        table: FieldTable,
        parameters: BindParameterView,
    ) -> None:
        """Pair the frozen table with the gated parameter view."""
        self._table = table
        self.parameters = parameters

    def __getattr__(self, name: str) -> object:
        """Delegate everything else to the field table."""
        return getattr(self._table, name)

    def __getitem__(self, name: str) -> FieldRecord:
        """Return the resolved row of one declared field."""
        return self._table[name]

    def __contains__(self, name: str) -> bool:
        """Whether ``name`` is a declared field."""
        return name in self._table

    def __iter__(self) -> Iterator[FieldRecord]:
        """Iterate the resolved rows, declaration order."""
        return iter(self._table)

    def __len__(self) -> int:
        """Return the number of declared fields."""
        return len(self._table)

    def __repr__(self) -> str:
        """Bind-table summary referencing the field table."""
        return f"<bind table over {self._table!r}>"


# ================================================================
#  Fingerprint (the restart fingerprint; 02_rules scope)
# ================================================================
@dataclass(frozen=True)
class Fingerprint:

    """
    The restart fingerprint: digest + diffable source record.

    Description
    -----------
    Hashes STRUCTURE, never leaves (02_rules): field declarations
    (names / declared patterns / bare interned spaces / lifecycles —
    never ``Layout`` or device topology), the module tuple (types +
    order), per-term treatments, stepper statics, and parameter
    *specs* (a Ramp's shape is structure, its endpoints are leaves).
    IC/state differences are deliberately invisible. The ``source``
    rows make mismatches human-diffable — ``SnapshotMismatchError``
    prints :meth:`diff`, never a silent reuse.

    Parameters
    ----------
    digest : str
        The structure-only hex digest.
    source : tuple[tuple[str, str], ...]
        The (key, token) rows the digest was computed over.
    """

    digest: str
    source: tuple[tuple[str, str], ...]

    def diff(self, other: Fingerprint) -> str:
        """
        Return the human-readable structural diff.

        Description
        -----------
        One line per differing row, in the style
        ``"stepper statics differ: cnab2 -> sbdf2"``; rows present
        on only one side are reported as such.

        Parameters
        ----------
        other : Fingerprint
            The fingerprint to compare against.

        Returns
        -------
        str
            The diff lines, or ``"fingerprints match"``.
        """
        mine = dict(self.source)
        theirs = dict(other.source)
        lines = []
        for key, token in self.source:
            if key not in theirs:
                lines.append(f"{key} only here: {token}")
            elif theirs[key] != token:
                lines.append(f"{key} differ: {token} -> "
                             f"{theirs[key]}")
        lines.extend(
            f"{key} only in other: {token}"
            for key, token in other.source if key not in mine)
        if not lines:
            return "fingerprints match"
        return "\n".join(lines)


# ================================================================
#  AssemblyRecord (the hashable static; the shared jit-cache key)
# ================================================================
#: composed step bodies, memoized by STRUCTURAL record equality —
#: identical re-assemblies (sweep members) share one entry, so the
#: composed callable itself never enters a jit closure
_STEP_FUNCTIONS: Final[dict[AssemblyRecord, Callable]] = {}


@dataclass(frozen=True, eq=False)
class AssemblyRecord:

    """
    Everything static the traced step depends on (jit-cache key).

    Description
    -----------
    The hashable static bundle keying the shared ``step_chunk``
    entry: ``__eq__``/``__hash__`` are STRUCTURAL (grid identity +
    the component tables' fingerprint tokens), so identical
    re-assemblies compare equal and sweep members share the compiled
    chunk. ``name`` is excluded from equality and hash. Construction
    runs the unhashable-static lint (an unhashable static would
    silently poison the jit cache).

    Parameters
    ----------
    grid : Grid
        Identity-hashed static (cache sharing requires the same
        grid object).
    field_table : FieldTable
        The resolved field table (step 1).
    binding_table : ParameterBindingTable
        The frozen binding table (step 2).
    remat_table : RematerializationTable
        The retained AUXILIARY defaults (step 5).
    schedule : Schedule
        The static kind-ordered schedule (step 5).
    stepper_statics : tuple
        The stepper's static identity (type + non-leaf attributes;
        ``dt`` and every provided leaf excluded).
    state_type : type
        The State vocabulary class (``VectorField`` fallback).
    extra_halo : HaloSpec | None
        The merged module ``extra_halo`` declarations.
    term_filter_token : str | None
        Variant provenance (the canonical filter token).
    module_types : tuple[str, ...]
        Module class names, tuple order (spec concretization: a
        term-free, field-free module must still distinguish
        records).
    parameter_specs : tuple[tuple[str, str], ...]
        Per-name value SPECS (scalar vs Ramp shape — structure per
        02_rules; spec concretization: carried on the record so the
        fingerprint stays a pure derivation).
    name : str | None
        Report/log attribution; excluded from ``__eq__``/``__hash__``
        (default: None).
    """

    grid: Grid
    field_table: FieldTable
    binding_table: ParameterBindingTable
    remat_table: RematerializationTable
    schedule: Schedule
    stepper_statics: tuple
    state_type: type
    extra_halo: HaloSpec | None
    term_filter_token: str | None
    module_types: tuple[str, ...] = ()
    parameter_specs: tuple[tuple[str, str], ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        """Run the unhashable-static lint (names the offender)."""
        _lint_hashable("stepper statics", self.stepper_statics)
        _lint_hashable("state_type", self.state_type)
        _lint_hashable("extra_halo", self.extra_halo)
        _lint_hashable("term_filter_token", self.term_filter_token)
        _lint_hashable("module_types", self.module_types)
        _lint_hashable("parameter_specs", self.parameter_specs)

    # ================================================================
    #  Structural identity
    # ================================================================
    def _token(self) -> tuple:
        """Return the structural token (grid handled separately)."""
        return (
            self.module_types,
            self.field_table.fingerprint_token(),
            self.binding_table.fingerprint_token(),
            self.remat_table.fingerprint_token(),
            self.schedule,
            self.stepper_statics,
            self.state_type,
            self.extra_halo,
            self.term_filter_token,
            self.parameter_specs,
        )

    def __eq__(self, other: object) -> bool:
        """Structural equality: grid identity + component tokens."""
        if not isinstance(other, AssemblyRecord):
            return NotImplemented
        return (self.grid is other.grid
                and self._token() == other._token())

    def __hash__(self) -> int:
        """Structural hash, matching ``__eq__`` (name excluded)."""
        return hash((id(self.grid), self._token()))

    # ================================================================
    #  Derived products
    # ================================================================
    def step_fn(self) -> Callable:
        """
        Return the composed step body, memoized keyed by ``self``.

        Description
        -----------
        The composed callable never enters a jit closure (a
        per-assembly closure would silently defeat the shared jit
        cache): ``assemble()`` seeds the memo under the record's
        STRUCTURAL key, so identical re-assemblies retrieve the one
        existing body.

        Returns
        -------
        Callable
            The composed step body ``(state, modules, ctx) ->
            (state, TendencySums)``.

        Raises
        ------
        AssemblyError
            If no step body is memoized under this record (records
            are produced by ``assemble()``, never rebuilt by hand).
        """
        try:
            return _STEP_FUNCTIONS[self]
        except KeyError:
            raise AssemblyError(
                "no composed step is memoized under this assembly "
                "record; records (and their step bodies) are "
                "produced by assemble(), never rebuilt by hand",
            ) from None

    def fingerprint(self) -> Fingerprint:
        """
        Compute the restart fingerprint (a pure record derivation).

        Returns
        -------
        Fingerprint
            The structure-only digest plus its diffable source.
        """
        rows: list[tuple[str, str]] = [
            ("modules", " -> ".join(self.module_types) or "none"),
        ]
        rows.extend(
            (f"field {name}", f"{pattern} on {space} [{lifecycle}]")
            for name, pattern, space, lifecycle
            in self.field_table.fingerprint_token())
        specs = dict(self.parameter_specs)
        rows.extend(
            (f"parameter {name}",
             f"slot={slot} attr={attr or '-'} "
             f"spec={specs.get(name, const_spec or '-')}")
            for name, slot, attr, const_spec
            in self.binding_table.fingerprint_token())
        for entry in self.schedule.entries:
            if entry.is_term:
                rows.append((f"term {entry.key}",
                             entry.treatment.name))
            else:
                rows.append((f"stage {entry.key}",
                             f"{entry.kind.name} "
                             f"(order={entry.order})"))
        rows.append(("stepper statics", repr(self.stepper_statics)))
        rows.append(("state type", self.state_type.__qualname__))
        if self.term_filter_token is not None:
            rows.append(("term filter", self.term_filter_token))
        source = tuple(rows)
        digest = hashlib.sha256(
            repr(source).encode("utf-8")).hexdigest()
        return Fingerprint(digest=digest, source=source)


def _lint_hashable(label: str, value: object) -> None:
    """
    Raise the unhashable-static lint, naming the offender.

    Parameters
    ----------
    label : str
        Attribution of the static being probed.
    value : object
        The static value.

    Raises
    ------
    AssemblyError
        If ``value`` (or a nested tuple item) is unhashable; the
        message names the innermost offending entry.
    """
    try:
        hash(value)
    except TypeError:
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                sub = f"{label}[{index}]"
                probe = item
                if (isinstance(item, tuple)
                        and len(item) == 2  # noqa: PLR2004
                        and isinstance(item[0], str)):
                    sub, probe = f"{label} {item[0]!r}", item[1]
                _lint_hashable(sub, probe)
        raise AssemblyError(
            f"assembly-record static {label} is unhashable "
            f"({type(value).__name__}: {value!r}); the record keys "
            "the shared jitted step, so every static must hash — "
            "declare array-valued attributes as dynamic leaves, or "
            "intern the object") from None


# ================================================================
#  AssemblyArtifacts (the wave-4.2 seam contract)
# ================================================================
@dataclass(frozen=True)
class AssemblyArtifacts:

    """
    What ``assemble()`` returns; what ``Model.__init__`` consumes.

    Description
    -----------
    A frozen host bundle. The ATTRIBUTE NAMES are the wave-4.2 seam
    contract — ``Model.__init__`` consumes exactly these names; do
    not rename.

    Parameters
    ----------
    field_table : FieldTable
        The resolved field table (step 1).
    binding_table : ParameterBindingTable
        The frozen binding table (step 2).
    remat_table : RematerializationTable
        The retained AUXILIARY defaults (step 5).
    composer : TendencyComposer
        The composer (its ``compose()`` product is also memoized on
        the record's ``step_fn`` seam).
    schedule : Schedule
        The static kind-ordered schedule.
    record : AssemblyRecord
        The hashable static bundle (jit-cache key).
    fingerprint : Fingerprint
        The restart fingerprint.
    report : AssemblyReport
        The printable assembly report.
    resharding : ReshardingReport
        The step-7 negotiation report (the model re-homes pre-built
        leaves with it).
    """

    field_table: FieldTable
    binding_table: ParameterBindingTable
    remat_table: RematerializationTable
    composer: TendencyComposer
    schedule: Schedule
    record: AssemblyRecord
    fingerprint: Fingerprint
    report: AssemblyReport
    resharding: ReshardingReport


# ================================================================
#  assemble() — the nine-step pipeline (steps 1-7 + 9)
# ================================================================
def assemble(
    *,
    grid: Grid,
    modules: tuple,
    time_stepper: object,
    state_type: type | None = None,
    name: str | None = None,
    term_filter: Callable | None = None,
) -> AssemblyArtifacts:
    """
    Run the nine-step assembly pipeline (model.md section 6.2).

    Description
    -----------
    Pure and deterministic in its inputs. The steps, in normative
    order: (1) collect declarations/references, resolve patterns
    through the grid's ``("declared_space", mesh)`` resolver rows,
    build the `FieldTable`; (2) build the `ParameterBindingTable`
    (the stepper joins as the ``fr.params.TIME_STEP`` provider);
    (3) the dispatch merge — resolve pattern keys through the step-1
    resolvers, ``grid.merge_overrides`` exactly once; (4)
    ``bind(table)`` in module order (the merged registry visible;
    bare time-dependent parameter reads gated); (5) collect terms +
    stages into the `TendencyComposer` (static checks live there),
    the re-materialization table, and the merged ``extra_halo``;
    (6) the composer dry run; (7) ``grid.negotiate(state_spaces=...,
    tendency=..., halo=...)`` + ``grid.freeze()`` — or, on an
    already-frozen grid, the verify path; (8) carry allocation is
    NOT run here (wave 4.2); (9) build the `AssemblyRecord`, the
    `Fingerprint`, and the `AssemblyReport`.

    Parameters
    ----------
    grid : Grid
        The assembly root (frozen after step 7).
    modules : tuple
        The module tuple (``fr.Module`` instances; duck-typed
        capability reads).
    time_stepper : object
        The time stepper — REQUIRED, no default exists.
    state_type : type | None, optional
        The State vocabulary class; None reads the module-supplied
        one (>1 provider is an error) and falls back to
        ``VectorField`` (default: None).
    name : str | None, optional
        Report/log attribution (default: None).
    term_filter : Callable | None, optional
        Variant term predicate ``(key, term) -> bool`` (2.8
        mechanics; declarations/stages never filtered)
        (default: None).

    Returns
    -------
    AssemblyArtifacts
        The frozen artifact bundle (the wave-4.2 seam).

    Raises
    ------
    AssemblyError
        And its subclasses, per the model-layer error registry.
    GridFrozenError
        Step 7 verify path: genuinely larger demands on a frozen
        grid ("assemble the most demanding model first").
    """
    modules = tuple(modules)
    frozen_before = grid.fingerprint is not None

    # -- step 1: fields ------------------------------------------
    declarations = _collect_declarations(modules)
    table = FieldTable(
        (FieldRecord.from_declaration(
            declaration, owner=slot,
            owner_type=type(modules[slot]).__qualname__, grid=grid)
         for slot, declaration in declarations),
        grid=grid)
    for slot, module in enumerate(modules):
        for reference in getattr(module, "field_references", ()):
            table.require(reference,
                          module=_module_label(slot, module))
    state_type = _resolve_state_type(modules, state_type)

    # -- step 2: parameters --------------------------------------
    binding_table = ParameterBindingTable.build(modules,
                                                time_stepper)

    # -- step 3: dispatch merge (before bind/dry-run/negotiate) --
    overrides = _collect_dispatch_overrides(modules, grid)
    if overrides:
        grid.merge_overrides(overrides)
    elif not frozen_before:
        # the exactly-once merge moment of a first, override-free
        # model (bakes Dispatched holes against the defaults)
        grid.merge_overrides({})

    # -- step 4: bind, module order ------------------------------
    bind_table = _BindTable(table, BindParameterView({
        str(entry.name): _read_leaf(entry, modules, time_stepper)
        for entry in binding_table}))
    for module in modules:
        bind = getattr(module, "bind", None)
        if callable(bind):
            bind(bind_table)

    # -- step 5: terms + stages, remat table, extra halo ---------
    terms = _collect_terms(modules)
    stages = _collect_stages(modules)
    remat_table = _build_remat_table(declarations, table, modules)
    extra_halo = _merged_extra_halo(modules)
    composer = TendencyComposer(
        field_table=table, modules=modules, terms=terms,
        stages=stages, time_stepper=time_stepper,
        binding_table=binding_table, term_filter=term_filter)
    schedule = composer.schedule

    # -- step 6: dry run -----------------------------------------
    # pass the assembly-time evaluated params so a term reading
    # ctx.params[...] resolves (an unbound name surfaces attributed
    # through the TermEvaluationError chain, not on an empty {})
    composer.dry_run(
        params=binding_table.eval_params(modules, time_stepper, 0.0))

    # -- step 7: negotiate + freeze (or the verify path) ---------
    resharding = _negotiate(grid, table, schedule, modules,
                            time_stepper, binding_table, extra_halo)
    grid.freeze()

    # -- step 8: carry allocation is wave 4.2 (not run here) -----

    # -- step 9: record, fingerprint, report ---------------------
    record = AssemblyRecord(
        grid=grid,
        field_table=table,
        binding_table=binding_table,
        remat_table=remat_table,
        schedule=schedule,
        stepper_statics=_stepper_statics(time_stepper,
                                         binding_table),
        state_type=state_type,
        extra_halo=extra_halo,
        term_filter_token=_term_filter_token(term_filter),
        module_types=tuple(type(module).__qualname__
                           for module in modules),
        parameter_specs=_parameter_specs(binding_table, modules,
                                         time_stepper),
        name=name)
    _STEP_FUNCTIONS.setdefault(record, composer.compose())
    fingerprint = record.fingerprint()
    report = _build_report(
        grid=grid, table=table, binding_table=binding_table,
        modules=modules, time_stepper=time_stepper,
        schedule=schedule, overrides=overrides,
        frozen_before=frozen_before, resharding=resharding,
        fingerprint=fingerprint, terms=terms,
        term_filter=term_filter, name=name)
    return AssemblyArtifacts(
        field_table=table, binding_table=binding_table,
        remat_table=remat_table, composer=composer,
        schedule=schedule, record=record, fingerprint=fingerprint,
        report=report, resharding=resharding)


# ================================================================
#  Step 1 helpers (fields, state type)
# ================================================================
def _collect_declarations(
    modules: tuple,
) -> tuple[tuple[int, FieldDeclaration], ...]:
    """Collect (slot, declaration) pairs, module tuple order."""
    return tuple(
        (slot, declaration)
        for slot, module in enumerate(modules)
        for declaration in getattr(module, "field_declarations",
                                   ()))


def _resolve_state_type(
    modules: tuple,
    state_type: type | None,
) -> type:
    """
    Resolve the State vocabulary class (D1.3 commitment 4).

    Description
    -----------
    The explicit ``Model(state_type=...)`` kwarg wins; otherwise the
    single module-supplied class; more than one distinct provider is
    an assembly error; the fallback is plain ``VectorField`` (a bare
    generic model loses nothing but sugar).
    """
    if state_type is not None:
        return state_type
    provided: list[tuple[str, type]] = []
    for slot, module in enumerate(modules):
        supplied = getattr(module, "state_type", None)
        if supplied is not None:
            provided.append((_module_label(slot, module), supplied))
    types = {cls for _, cls in provided}
    if len(types) > 1:
        owners = ", ".join(label for label, _ in provided)
        raise AssemblyError(
            f"more than one module supplies a state_type ({owners});"
            " exactly one dynamical core publishes the vocabulary "
            "class — pass Model(state_type=...) to override")
    if provided:
        return provided[0][1]
    return VectorField


# ================================================================
#  Step 3 helpers (dispatch merge)
# ================================================================
def _collect_dispatch_overrides(
    modules: tuple,
    grid: Grid,
) -> dict[str, dict]:
    """
    Collect per-module dispatch overrides, pattern keys resolved.

    Description
    -----------
    Builds the per-module form ``{module label: {key: op}}`` that
    ``grid.merge_overrides`` consumes (same resolved key from two
    modules raises ``DispatchCollisionError`` naming both).
    ``(kind, SpacePattern)`` keys are resolved through the step-1
    ``("declared_space", mesh)`` resolvers into ``(kind, space)``;
    a ``SpaceRule`` key is rejected (rules are identity-hashed
    behavior, never dispatch keys).
    """
    collected: dict[str, dict] = {}
    for slot, module in enumerate(modules):
        entries = getattr(module, "dispatch", None)
        if not entries:
            continue
        label = _module_label(slot, module)
        collected[label] = {
            _resolve_dispatch_key(key, grid, label): op
            for key, op in entries.items()}
    return collected


def _resolve_dispatch_key(
    key: object,
    grid: Grid,
    label: str,
) -> object:
    """Resolve one override key's pattern component (if any)."""
    if isinstance(key, tuple) and len(key) == 2:  # noqa: PLR2004
        kind, space = key
        if isinstance(space, SpaceRule):
            raise AssemblyError(
                f"{label}: dispatch key {key!r} carries a SpaceRule;"
                " rules are identity-hashed behavior and never "
                "dispatch keys — use a SpacePattern or the resolved "
                "space")
        if isinstance(space, SpacePattern):
            return (kind, space.resolve(grid))
    return key


# ================================================================
#  Step 5 helpers (collection)
# ================================================================
def _collect_terms(
    modules: tuple,
) -> tuple[tuple[int, TendencyTerm], ...]:
    """Collect (slot, term) pairs: module order, then declaration."""
    pairs: list[tuple[int, TendencyTerm]] = []
    for slot, module in enumerate(modules):
        collect = getattr(module, "tendency_terms", None)
        if callable(collect):
            pairs.extend((slot, term) for term in collect())
    return tuple(pairs)


def _collect_stages(
    modules: tuple,
) -> tuple[tuple[int, Stage], ...]:
    """Collect (slot, stage) pairs: module order, then declaration."""
    pairs: list[tuple[int, Stage]] = []
    for slot, module in enumerate(modules):
        collect = getattr(module, "collected_stages", None)
        if callable(collect):
            pairs.extend((slot, stage) for stage in collect())
    return tuple(pairs)


def _build_remat_table(
    declarations: tuple[tuple[int, FieldDeclaration], ...],
    table: FieldTable,
    modules: tuple,
) -> RematerializationTable:
    """Retain the AUXILIARY declaration defaults (the D1.1 soften)."""
    return RematerializationTable(tuple(
        RematerializationEntry.from_declaration(
            declaration, owner=slot,
            space=table[declaration.name].space,
            owner_instance=modules[slot])
        for slot, declaration in declarations
        if declaration.lifecycle is Lifecycle.AUXILIARY))


def _merged_extra_halo(modules: tuple) -> HaloSpec | None:
    """Merge (max) the modules' declared ``extra_halo`` specs."""
    merged: HaloSpec | None = None
    for module in modules:
        spec = getattr(module, "extra_halo", None)
        if spec is None:
            continue
        merged = spec if merged is None else merged.merge_max(spec)
    return merged


# ================================================================
#  Step 7 helpers (negotiate + freeze / the verify path)
# ================================================================
def _negotiate(
    grid: Grid,
    table: FieldTable,
    schedule: Schedule,
    modules: tuple,
    time_stepper: object,
    binding_table: ParameterBindingTable,
    extra_halo: HaloSpec | None,
) -> ReshardingReport:
    """
    Run assembly step 7 (or the frozen-grid verify path).

    Description
    -----------
    ``grid.negotiate`` receives the NAME-KEYED state spaces (the
    tracer state must carry component names — attribution and the
    FieldTable cross-checks key on them), the composed body adapted
    to the tracer calling convention, and the merged halo demand
    (combined ``merge_max`` with the trace by the grid). A field-free
    composition negotiates with an explicit zero-or-extra halo (no
    tendency to trace, and the unscoped registry maximum must not
    leak into the demand).

    The halo demand combines the module-declared ``extra_halo`` (the
    genuine raw-``.data`` bypasses — the spectral projection) with the
    ``derived_halo`` auto-read off every linear term's block operators
    (R13): a linear module's coefficient scaling would raise on a
    tracer (V-N2), so its terms are EXEMPT from the numeric trace and
    their exact stencil demand is supplied here instead — tighter than
    the old all-axes hand-declaration, never smaller.
    """
    spaces = {record.name: record.space for record in table}
    if not spaces:
        halo = (HaloSpec.zero(grid.names) if extra_halo is None
                else extra_halo)
        return grid.negotiate(halo=halo)
    exempt = frozenset(
        slot for slot, module in enumerate(modules)
        if getattr(module, "extra_halo", None) is not None)
    derived_halo, exempt_terms = _linear_terms_halo(
        modules, spaces, grid.dispatch)
    tendency = _tracer_tendency(
        schedule, modules, exempt, exempt_terms,
        _tracer_params(binding_table, modules, time_stepper))
    return grid.negotiate(
        state_spaces=_StateSpaceMapping(spaces),
        tendency=tendency,
        halo=_merge_halo(extra_halo, derived_halo))


def _linear_terms_halo(
    modules: tuple,
    spaces: Mapping[str, SpaceLike],
    registry: object,
) -> tuple[HaloSpec | None, frozenset[str]]:
    """
    Derive the linear terms' halo and their exempt entry keys (R13).

    Description
    -----------
    Each ``linear`` term with declared ``blocks`` contributes the
    ghost-width demand of its block operators (coefficient-free — the
    scaling is pointwise, halo 0) through ``linear_blocks_halo``, and
    is EXEMPT from the numeric halo trace: its coefficient scaling
    would raise on a tracer (the V-N2 raw-``.data`` gate). The exempt
    keys mirror the composer's ``"Module/term"`` attribution so
    ``_tracer_tendency`` can skip exactly those scheduled entries.
    """
    # local import breaks the assembly <-> linear_blocks import cycle
    from fridom.framework2.model.linear_blocks import (  # noqa: PLC0415
        linear_blocks_halo,
    )
    derived: HaloSpec | None = None
    keys: set[str] = set()
    for slot, term in _collect_terms(modules):
        if not getattr(term, "linear", False) or not term.blocks:
            continue
        keys.add(f"{type(modules[slot]).__name__}/{term.name}")
        spec = linear_blocks_halo(term.blocks, spaces, registry)
        derived = spec if derived is None else derived.merge_max(spec)
    return derived, frozenset(keys)


def _merge_halo(
    left: HaloSpec | None, right: HaloSpec | None,
) -> HaloSpec | None:
    """Pointwise-max two optional halo specs (None is the identity)."""
    if left is None:
        return right
    if right is None:
        return left
    return left.merge_max(right)


def _tracer_params(
    binding_table: ParameterBindingTable,
    modules: tuple,
    time_stepper: object,
) -> dict[str, object]:
    """
    Evaluate the bound parameters at t=0 for the halo trace.

    Description
    -----------
    The tracer state understands plain Python scalars (its
    arithmetic treats everything else as a field operand), so leaf
    values are demoted to scalars where possible; non-scalar leaves
    pass through unchanged.
    """
    resolved = binding_table.eval_params(modules, time_stepper, 0.0)
    return {name: _as_scalar(resolved[name]) for name in resolved}


def _as_scalar(value: object) -> object:
    """Demote a 0-d numeric leaf to a Python scalar (best effort)."""
    if isinstance(value, int | float | complex):
        return value
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return complex(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value


def _tracer_tendency(
    schedule: Schedule,
    modules: tuple,
    exempt: frozenset[int],
    exempt_terms: frozenset[str],
    params: Mapping[str, object],
) -> Callable[[object], object]:
    """
    Adapt the composed body to the halo tracer's calling convention.

    Description
    -----------
    ``trace_halo`` calls ``tendency(state)`` with a name-keyed
    ``VectorTracer``; the composed body takes ``(state, modules,
    ctx)``. This adapter closes over the assembly modules and a
    tracer-safe context (scalar clock/params — ``StepContext`` is
    all-scalar by design, so the tracer needs zero ctx mimicry) and
    mirrors the composed body's schedule walk: SELF_UPDATE ->
    DIAGNOSE -> EXPLICIT terms -> CONSTRAINT. Hooks of ``exempt``
    (``extra_halo``-declaring) modules are skipped — their declared
    spec substitutes (V-N2), and the second dry-run mode over real
    zero fields already validated them. Linear terms in
    ``exempt_terms`` are likewise skipped (their coefficient scaling
    would raise on a tracer); the R13 ``derived_halo`` supplies their
    stencil demand instead.
    """
    def tendency(state: object) -> object:
        """Walk the schedule once over the tracer state."""
        ctx = StepContext(params=params, clock=0.0, dt=1.0,
                          stage_dt=1.0)
        for kind in (StageKind.SELF_UPDATE, StageKind.DIAGNOSE):
            state = _trace_stage_kind(schedule, kind, modules,
                                      exempt, state, ctx)
        for entry in schedule.kind_entries(None):
            if (entry.slot in exempt
                    or entry.key in exempt_terms
                    or entry.treatment is not Treatment.EXPLICIT):
                continue
            evaluate_entry(entry, modules[entry.slot], state, ctx)
        return _trace_stage_kind(schedule, StageKind.CONSTRAINT,
                                 modules, exempt, state, ctx)

    return tendency


def _trace_stage_kind(
    schedule: Schedule,
    kind: StageKind,
    modules: tuple,
    exempt: frozenset[int],
    state: object,
    ctx: StepContext,
) -> object:
    """Trace one stage kind (replace-applied), skipping exempts."""
    for entry in schedule.kind_entries(kind):
        if entry.slot in exempt:
            continue
        result = evaluate_entry(entry, modules[entry.slot], state,
                                ctx)
        state = apply_replace(entry, state, result)
    return state


# ================================================================
#  Step 9 helpers (record + fingerprint inputs)
# ================================================================
def _stepper_statics(
    time_stepper: object,
    binding_table: ParameterBindingTable,
) -> tuple:
    """
    Derive the stepper's static identity (dt and leaves excluded).

    Description
    -----------
    Type name plus every instance attribute that is neither a
    declared dynamic leaf (``dynamic_jax_attrs``), nor a provided
    parameter attribute (the binding table's stepper rows — this is
    what excludes ``dt``), nor an equality-exempt host observer.
    Hashability is enforced by the record's unhashable-static lint.
    """
    token = getattr(time_stepper, "fingerprint_token", None)
    if callable(token):
        return token()
    provided = {entry.attr for entry in binding_table
                if entry.slot == "stepper"}
    dynamic = set(getattr(time_stepper, "dynamic_jax_attrs", ())
                  or ())
    ignored = set(getattr(time_stepper, "_eq_ignored_attrs", ())
                  or ())
    excluded = provided | dynamic | ignored
    statics = tuple(
        (attr, value)
        for attr, value in sorted(vars(time_stepper).items())
        if attr not in excluded)
    return (type(time_stepper).__qualname__, statics)


def _parameter_specs(
    binding_table: ParameterBindingTable,
    modules: tuple,
    time_stepper: object,
) -> tuple[tuple[str, str], ...]:
    """
    Derive per-name value SPECS (02_rules: shape, never leaves).

    Description
    -----------
    A scalar <-> Ramp swap is structure (a treedef change); a Ramp's
    curve is its static shape, its endpoints are leaves and stay
    invisible.
    """
    return tuple(
        (str(entry.name),
         _value_spec(_read_leaf(entry, modules, time_stepper)))
        for entry in binding_table)


def _value_spec(value: object) -> str:
    """Spell one live leaf's structural spec."""
    if isinstance(value, TimeDependent):
        curve = getattr(value, "_curve_spec", None)
        shape = (getattr(curve, "__name__", str(curve))
                 if curve is not None else "")
        return (f"{type(value).__name__}({shape})" if shape
                else type(value).__name__)
    return type(value).__name__


def _term_filter_token(term_filter: Callable | None) -> str | None:
    """
    Derive the canonical variant-filter token (fingerprint join).

    Description
    -----------
    Spec concretization: a predicate exposing ``token`` (the 2.8
    term-predicate algebra) contributes it verbatim; otherwise the
    qualified name stands in.
    """
    if term_filter is None:
        return None
    token = getattr(term_filter, "token", None)
    if token is not None:
        return str(token)
    return getattr(term_filter, "__qualname__",
                   type(term_filter).__qualname__)


# ================================================================
#  Step 9 helpers (the report sections)
# ================================================================
def _build_report(
    *,
    grid: Grid,
    table: FieldTable,
    binding_table: ParameterBindingTable,
    modules: tuple,
    time_stepper: object,
    schedule: Schedule,
    overrides: dict[str, dict],
    frozen_before: bool,
    resharding: ReshardingReport,
    fingerprint: Fingerprint,
    terms: tuple[tuple[int, TendencyTerm], ...],
    term_filter: Callable | None,
    name: str | None,
) -> AssemblyReport:
    """Compose the eight report sections (model.md section 3)."""
    return AssemblyReport({
        "header": _header_section(grid, time_stepper, modules,
                                  fingerprint, name),
        "fields": _fields_section(table),
        "parameters": _parameters_section(binding_table, modules),
        "dispatch": _dispatch_section(overrides, frozen_before),
        "schedule": schedule.describe(),
        "halo": _halo_section(grid, resharding),
        "lint": _lint_section(table, terms, term_filter),
        "run_start": RUN_START_PLACEHOLDER,
    })


def _header_section(
    grid: Grid,
    time_stepper: object,
    modules: tuple,
    fingerprint: Fingerprint,
    name: str | None,
) -> str:
    """Header: name / grid / stepper / modules / digest."""
    title = "model assembly" + (f" {name!r}" if name else "")
    meshes = " * ".join(repr(mesh) for mesh in grid.factors)
    module_names = ", ".join(
        type(module).__qualname__ for module in modules) or "none"
    return "\n".join((
        title,
        f"grid: {meshes}",
        f"time stepper: {type(time_stepper).__qualname__}",
        f"modules: {module_names}",
        f"fingerprint: {fingerprint.digest}",
    ))


def _fields_section(table: FieldTable) -> str:
    """Fields: name -> owner -> pattern -> space -> lifecycle."""
    lines = []
    for record in table:
        roles = ", ".join(sorted(repr(role)
                                 for role in record.roles)) or "-"
        lines.append(
            f"{record.name}: {record.pattern!r} -> {record.space!r}"
            f" [{record.lifecycle.name}] owner="
            f"{record.owner_type} (modules[{record.owner}]) "
            f"roles: {roles}")
    if not lines:
        lines.append("no declared fields (legal, CS-13)")
    writable = ", ".join(table.host_writable) or "none"
    lines.append(f"host-writable: {writable}")
    return "\n".join(lines)


def _parameters_section(
    binding_table: ParameterBindingTable,
    modules: tuple,
) -> str:
    """Parameters: the binding table; identity defaults listed."""
    lines = []
    identity: list[str] = []
    for entry in binding_table:
        name = str(entry.name)
        if entry.slot is None:
            lines.append(
                f"{name} = {entry.value!r} (identity default)")
            identity.append(name)
        elif entry.slot == "stepper":
            lines.append(f"{name} <- the time stepper "
                         f".{entry.attr}")
        else:
            owner = type(modules[entry.slot]).__qualname__
            lines.append(f"{name} <- modules[{entry.slot}] "
                         f"({owner}).{entry.attr}")
    if not lines:
        lines.append("no bound parameters")
    defaults = ", ".join(identity) or "none"
    lines.append(f"identity defaults in effect: {defaults}")
    return "\n".join(lines)


def _dispatch_section(
    overrides: dict[str, dict],
    frozen_before: bool,
) -> str:
    """Dispatch: merged overrides (or the verify-path outcome)."""
    lines = []
    if frozen_before:
        lines.append("frozen grid: verify path (no merge ran)")
    for label, entries in overrides.items():
        lines.extend(f"{label}: {key!r}" for key in entries)
    if not overrides:
        lines.append("no module dispatch overrides")
    return "\n".join(lines)


def _halo_section(grid: Grid, resharding: ReshardingReport) -> str:
    """Halo/layout: the negotiated widths and layout outcome."""
    record = grid.fingerprint
    widths = (dict(record.halo.widths) if record is not None
              else "not negotiated")
    return "\n".join((
        f"negotiated halo: {widths}",
        f"default layout: {resharding.new!r}",
        f"layout changed by negotiation: {resharding.changed}",
    ))


def _lint_section(
    table: FieldTable,
    terms: tuple[tuple[int, TendencyTerm], ...],
    term_filter: Callable | None,
) -> str:
    """Lint: aggregated warnings (untransported ADVECTED, filter)."""
    lines = []
    advected = set(table.select(ADVECTED))
    transported: set[str] = set()
    for _, term in terms:
        transported.update(term.transports)
    untransported = sorted(advected - transported)
    if untransported:
        lines.append(
            "ADVECTED but transported by no term: "
            + ", ".join(untransported))
    if term_filter is not None:
        lines.append(
            "variant term filter active: the coverage lint is "
            "downgraded to a warning")
    if not lines:
        lines.append("none")
    return "\n".join(lines)
