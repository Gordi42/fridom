"""
The resolved field table.

Description
-----------
Owning class spec: ``notes/framework2/model/classes/model.md``
(sections "FieldTable (+ FieldRecord)" and "VelocitySelector").
`FieldRecord` is one resolved declaration row (owner, declared
pattern, resolved bare space, lifecycle, roles); `FieldTable` is the
Model-owned, declaration-ordered resolution table built at assembly
step 1 — frozen after construction and hashable (it joins the static
``AssemblyRecord``), the bind-time query surface (role selections
resolve once, into static tuples), and the mechanism behind the
model-mediated ``state.prognostic`` read (`FieldTable.subset` — the
adopted spelling; the FieldTable-as-state-aux proposal was rejected
2026-07-08). `VelocitySelector` resolves the Velocity role family
once at bind time (directional/transverse/prognostic splits per
V-N1/V-H2). Space resolution happens at table build via the declared
pattern's ``resolve(grid)`` against the grid-level
``("declared_space", mesh)`` resolver rows.
"""
# Wave 3 A: FieldRecord, FieldTable (+ subset), VelocitySelector
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fridom.framework2.model.declarations import Lifecycle
from fridom.framework2.model.errors import (
    AssemblyError,
    FieldCollisionError,
    MissingFieldError,
)
from fridom.framework2.model.roles import Role, Velocity

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator

    from fridom.framework2.grid.fields.metadata import FieldMetadata
    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import (
        TensorProductSpace,
    )
    from fridom.framework2.model.declarations import (
        FieldDeclaration,
        FieldReference,
    )
    from fridom.framework2.model.space_patterns import (
        SpacePattern,
        SpaceRule,
    )


# ================================================================
#  FieldRecord (one resolved declaration row)
# ================================================================
@dataclass(frozen=True)
class FieldRecord:

    """
    One resolved declaration row of the field table.

    Description
    -----------
    Everything *structural* a declaration contributes, after space
    resolution: the declarations themselves are transient assembly
    inputs (only AUXILIARY ``default=`` closures survive, in the
    re-materialization table). The ``name -> owner -> pattern ->
    space`` row is the report's field section and the
    typo'd-coordinate mitigation.

    Parameters
    ----------
    name : str
        Component key; the collision unit (dot-free, enforced
        upstream at declaration).
    owner : int
        The owning module's tuple index.
    owner_type : str
        Qualified class name of the owner (report/fingerprint).
    pattern : SpacePattern | SpaceRule
        The space descriptor as declared (the fingerprint uses it).
    space : TensorProductSpace
        The resolved bare interned space (pre-layout).
    lifecycle : Lifecycle
        Who advances the field.
    roles : frozenset[Role]
        The opt-in consumer tags.
    host_writable : bool
        Owner consent for host ``set_aux`` writes (CS-1/CS-2).
    metadata : FieldMetadata
        The grid-layer annotation that survives allocation.
    """

    name: str
    owner: int
    owner_type: str
    pattern: SpacePattern | SpaceRule
    space: TensorProductSpace
    lifecycle: Lifecycle
    roles: frozenset[Role]
    host_writable: bool
    metadata: FieldMetadata

    @classmethod
    def from_declaration(
        cls,
        declaration: FieldDeclaration,
        *,
        owner: int,
        owner_type: str,
        grid: Grid,
    ) -> FieldRecord:
        """
        Resolve one declaration against a grid (table-build seam).

        Description
        -----------
        Space resolution happens here: the declared pattern/rule is
        resolved through the grid-level ``("declared_space", mesh)``
        resolver rows into the bare interned space. The declaration's
        annotation is folded into the surviving `FieldMetadata`.

        Parameters
        ----------
        declaration : FieldDeclaration
            The module's declaration (validated at construction).
        owner : int
            The owning module's tuple index.
        owner_type : str
            Qualified class name of the owner.
        grid : Grid
            The grid whose resolver rows bind the pattern.

        Returns
        -------
        FieldRecord
            The resolved row.
        """
        return cls(
            name=declaration.name,
            owner=owner,
            owner_type=owner_type,
            pattern=declaration.space,
            space=declaration.space.resolve(grid),
            lifecycle=declaration.lifecycle,
            roles=declaration.roles,
            host_writable=declaration.host_writable,
            metadata=declaration.field_metadata())

    def fingerprint_token(self) -> tuple:
        """
        Return the row's restart-fingerprint contribution.

        Description
        -----------
        02_rules scope: name, declared pattern, resolved **bare**
        space, and lifecycle — never ``Layout`` or device topology.
        A `SpaceRule` pattern needs no token of its own (the
        resolved bare space carries the structure).

        Returns
        -------
        tuple
            Human-diffable ``(name, pattern, space, lifecycle)``
            string rows.
        """
        return (self.name, repr(self.pattern), repr(self.space),
                self.lifecycle.name)


# ================================================================
#  VelocitySelector (the Velocity family, resolved once)
# ================================================================
@dataclass(frozen=True)
class VelocitySelector:

    """
    The Velocity role family, resolved once at bind time.

    Description
    -----------
    Labels are opaque keys, never derived from spaces (B-grid
    ``u``/``v`` share one interned space; A-grids carry no signal).
    On tensor-product grids labels are coordinate names, and a label
    absent from the grid's names marks a transverse (slaved)
    component (V-N1): excluded from divergence/gradient/flux
    *directions* while remaining a full family member for friction,
    CFL, energy, and advection. Role-driven write-targeting
    intersects PROGNOSTIC automatically (the diagnosed ``w`` has no
    momentum equation — V-H2).

    Parameters
    ----------
    components : tuple[str, ...]
        All Velocity-role fields, declaration order (PROGNOSTIC and
        DIAGNOSTIC — reads span both).
    labels : tuple[tuple[str, str], ...]
        The ``(name, component label)`` pairs, declaration order.
    directional : tuple[str, ...]
        Fields whose labels match grid coordinate names: they enter
        divergence/gradient/flux directions.
    transverse : tuple[str, ...]
        Fields whose labels match no coordinate (slaved; their
        directional derivative is zero by construction).
    prognostic : tuple[str, ...]
        The write-target subset (friction targets this).
    """

    components: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]
    directional: tuple[str, ...]
    transverse: tuple[str, ...]
    prognostic: tuple[str, ...]

    def label_of(self, name: str) -> str:
        """
        Return the component label of one family member.

        Parameters
        ----------
        name : str
            A Velocity-family field name.

        Returns
        -------
        str
            The declared component label.

        Raises
        ------
        ValueError
            If ``name`` is not in the family.
        """
        for member, label in self.labels:
            if member == name:
                return label
        raise ValueError(
            f"field {name!r} is not in the velocity family; "
            f"members: {self.components}")


# ================================================================
#  FieldTable (the bind-time query surface)
# ================================================================
class FieldTable:

    """
    Declaration-ordered resolution table; frozen and hashable.

    Description
    -----------
    Built at assembly step 1 and locked before halo tracing and
    negotiation (MOM6-style); declaration order is component order.
    What ``bind(table)`` receives: by-name access, lifecycle splits,
    by-role selection (open sets, zero matches are a no-op), the
    Velocity-family selector, and the lifecycle-subset view backing
    the model-mediated ``state.prognostic`` read. Empty-PROGNOSTIC
    compositions are legal (CS-13). Construction discharges the
    one-owner-per-name check (`FieldCollisionError` naming **both**
    modules); unsatisfied references raise through :meth:`require`
    with the reference hint.

    Parameters
    ----------
    records : Iterable[FieldRecord]
        The resolved rows, in declaration order.
    grid : Grid | None, optional
        The grid the rows were resolved against; carried for
        downstream field materialization (the composer's dry run,
        allocation) and excluded from equality/hash — the resolved
        spaces already pin it structurally (default: None).
    """

    __slots__ = ("_by_name", "_grid", "_records")

    def __init__(self, records: Iterable[FieldRecord],
                 grid: Grid | None = None) -> None:
        """Validate the rows, check collisions, and freeze."""
        records = tuple(records)
        object.__setattr__(self, "_grid", grid)
        by_name: dict[str, FieldRecord] = {}
        for record in records:
            if not isinstance(record, FieldRecord):
                raise TypeError(
                    f"FieldTable rows are FieldRecords, got "
                    f"{record!r}")
            other = by_name.get(record.name)
            if other is not None:
                raise FieldCollisionError(
                    f"field {record.name!r} is declared twice: by "
                    f"{other.owner_type} (modules[{other.owner}]) "
                    f"and by {record.owner_type} "
                    f"(modules[{record.owner}]); declarations are "
                    "never silently merged — exactly one module "
                    "owns a field name")
            by_name[record.name] = record
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_by_name", by_name)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation: the table is frozen after step 1."""
        raise AttributeError(
            "FieldTable is frozen after construction (locked before "
            "halo tracing/negotiation); build a new table instead")

    # ================================================================
    #  Mapping-style access
    # ================================================================
    @property
    def grid(self) -> Grid | None:
        """The grid the rows were resolved against (may be None)."""
        return self._grid

    @property
    def names(self) -> tuple[str, ...]:
        """All declared field names, declaration order."""
        return tuple(record.name for record in self._records)

    def __getitem__(self, name: str) -> FieldRecord:
        """
        Return the resolved row of one declared field.

        Parameters
        ----------
        name : str
            The field name.

        Returns
        -------
        FieldRecord
            The resolved row.

        Raises
        ------
        MissingFieldError
            If no module declared ``name``.
        """
        try:
            return self._by_name[name]
        except KeyError:
            declared = ", ".join(self.names) or "none"
            raise MissingFieldError(
                f"no module declares the field {name!r}; declared "
                f"fields: {declared}") from None

    def __contains__(self, name: str) -> bool:
        """Whether ``name`` is a declared field."""
        return name in self._by_name

    def __iter__(self) -> Iterator[FieldRecord]:
        """Iterate the resolved rows, declaration order."""
        return iter(self._records)

    def __len__(self) -> int:
        """Return the number of declared fields."""
        return len(self._records)

    def require(
        self,
        reference: FieldReference,
        *,
        module: str = "",
    ) -> FieldRecord:
        """
        Discharge one ``FieldReference`` (assembly step 1 check).

        Parameters
        ----------
        reference : FieldReference
            The consumer's claim on a field it does not own.
        module : str, optional
            The requiring module's name, for attribution
            (default: "").

        Returns
        -------
        FieldRecord
            The satisfying row.

        Raises
        ------
        MissingFieldError
            If the referenced field is not declared; the message
            carries the reference hint and is attributed to the
            requiring module.
        """
        record = self._by_name.get(reference.name)
        if record is None:
            who = module or "a module"
            hint = f" (hint: {reference.hint})" if reference.hint \
                else ""
            declared = ", ".join(self.names) or "none"
            raise MissingFieldError(
                f"{who} references the field {reference.name!r}, "
                f"which no module declares{hint}; declared fields: "
                f"{declared}")
        return record

    # ================================================================
    #  Lifecycle splits
    # ================================================================
    def _lifecycle_names(
        self, lifecycle: Lifecycle,
    ) -> tuple[str, ...]:
        """Names of one lifecycle, declaration order."""
        return tuple(record.name for record in self._records
                     if record.lifecycle is lifecycle)

    @property
    def prognostic(self) -> tuple[str, ...]:
        """PROGNOSTIC names, declaration order (may be empty, CS-13)."""
        return self._lifecycle_names(Lifecycle.PROGNOSTIC)

    @property
    def auxiliary(self) -> tuple[str, ...]:
        """AUXILIARY names, declaration order."""
        return self._lifecycle_names(Lifecycle.AUXILIARY)

    @property
    def diagnostic(self) -> tuple[str, ...]:
        """DIAGNOSTIC names, declaration order."""
        return self._lifecycle_names(Lifecycle.DIAGNOSTIC)

    @property
    def host_writable(self) -> tuple[str, ...]:
        """Consented names (``model.report`` listing; set_aux gate)."""
        return tuple(record.name for record in self._records
                     if record.host_writable)

    # ================================================================
    #  Role queries (bind-time; resolved into static tuples)
    # ================================================================
    def select(self, role: Role | type[Role]) -> tuple[str, ...]:
        """
        By-role query: open set, may match zero fields.

        Description
        -----------
        Family matching is class-vs-instance: ``select(Velocity)``
        (the class) matches any component; ``select(Velocity("x"))``
        exactly one. Zero matches are a no-op, never an error — the
        reference/role distinction.

        Parameters
        ----------
        role : Role | type[Role]
            A role instance (exact match) or a Role subclass
            (family match).

        Returns
        -------
        tuple[str, ...]
            The matching field names, declaration order.
        """
        if isinstance(role, type):
            if not issubclass(role, Role):
                raise TypeError(
                    f"select() takes a Role instance or a Role "
                    f"subclass, got {role!r}")
            return tuple(
                record.name for record in self._records
                if any(isinstance(tag, role)
                       for tag in record.roles))
        if not isinstance(role, Role):
            raise TypeError(
                f"select() takes a Role instance or a Role "
                f"subclass, got {role!r}")
        return tuple(record.name for record in self._records
                     if role in record.roles)

    def velocity(self) -> VelocitySelector:
        """
        Resolve the Velocity family (replaces positional slices).

        Returns
        -------
        VelocitySelector
            The family splits, resolved once into static tuples.

        Raises
        ------
        AssemblyError
            If the family is ambiguous: one field carrying two
            Velocity labels, or two fields carrying the same label
            (the message names both owners).
        """
        labels: list[tuple[str, str]] = []
        by_label: dict[str, FieldRecord] = {}
        for record in self._records:
            tags = sorted(
                tag.component for tag in record.roles
                if isinstance(tag, Velocity))
            if not tags:
                continue
            if len(tags) > 1:
                raise AssemblyError(
                    f"velocity family is ambiguous: field "
                    f"{record.name!r} ({record.owner_type}) carries "
                    f"multiple Velocity labels {tuple(tags)}; one "
                    "component, one label")
            label = tags[0]
            other = by_label.get(label)
            if other is not None:
                raise AssemblyError(
                    f"velocity family is ambiguous: the label "
                    f"{label!r} is declared on both {other.name!r} "
                    f"({other.owner_type}) and {record.name!r} "
                    f"({record.owner_type}); labels are distinct "
                    "across velocity declarations")
            by_label[label] = record
            labels.append((record.name, label))
        coordinates = self._coordinate_names()
        return VelocitySelector(
            components=tuple(name for name, _ in labels),
            labels=tuple(labels),
            directional=tuple(name for name, label in labels
                              if label in coordinates),
            transverse=tuple(name for name, label in labels
                             if label not in coordinates),
            prognostic=tuple(
                name for name, _ in labels
                if self._by_name[name].lifecycle
                is Lifecycle.PROGNOSTIC))

    def _coordinate_names(self) -> frozenset[str]:
        """Grid coordinate names, from the resolved bare spaces."""
        return frozenset(
            name for record in self._records
            for name in record.space.names)

    # ================================================================
    #  Lifecycle-subset view (backs the model-mediated read)
    # ================================================================
    def subset(
        self,
        state: VectorField,
        lifecycle: Lifecycle,
    ) -> VectorField:
        """
        Lifecycle-subset view of an assembled state.

        Description
        -----------
        Backs the model-mediated ``state.prognostic``-style read
        (the adopted spelling, 2026-07-08; Tier-2 read-back): the
        model, which owns the table, does the subsetting — plain
        states carry no table. Components are returned in
        declaration order, preserving the state's class.

        Parameters
        ----------
        state : VectorField
            An assembled state carrying the declared components.
        lifecycle : Lifecycle
            The lifecycle to select.

        Returns
        -------
        VectorField
            The subset, same class as ``state``.

        Raises
        ------
        ValueError
            If this table declares no field of ``lifecycle`` (an
            empty vector has no components).
        MissingFieldError
            If ``state`` lacks a declared component (it is not a
            state assembled from this table).
        """
        names = self._lifecycle_names(lifecycle)
        if not names:
            raise ValueError(
                f"this composition declares no {lifecycle.name} "
                "field (legal, CS-13), so there is no "
                f"{lifecycle.name} subset to build")
        missing = tuple(name for name in names
                        if name not in state)
        if missing:
            raise MissingFieldError(
                f"state lacks the declared {lifecycle.name} "
                f"component(s) {missing}; lifecycle subsets are "
                "views of states assembled from this table "
                f"(components present: "
                f"{', '.join(state.component_names)})")
        return type(state)({name: state[name] for name in names})

    # ================================================================
    #  Hashability (the table joins the static AssemblyRecord)
    # ================================================================
    def fingerprint_token(self) -> tuple:
        """
        Return the table's restart-fingerprint contribution.

        Returns
        -------
        tuple
            One human-diffable token row per record.
        """
        return tuple(record.fingerprint_token()
                     for record in self._records)

    def __eq__(self, other: object) -> bool:
        """Structural equality over the resolved rows."""
        if not isinstance(other, FieldTable):
            return NotImplemented
        return self._records == other._records

    def __hash__(self) -> int:
        """Structural hash over the resolved rows."""
        return hash(self._records)

    def __repr__(self) -> str:
        """Summary repr: field count and lifecycle split."""
        return (f"FieldTable({len(self._records)} fields: "
                f"{len(self.prognostic)} prognostic, "
                f"{len(self.auxiliary)} auxiliary, "
                f"{len(self.diagnostic)} diagnostic)")
