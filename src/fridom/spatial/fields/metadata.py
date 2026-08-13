"""
``FieldMetadata``: annotation metadata for fields.

Description
-----------
Owning class doc: ``design/specs/grid/classes/fields.md``. Pure
annotation for naming and I/O (name, units, nc-attrs) with no
discretization content: metadata never influences dispatch, dtype,
shape, or algebra. Frozen and hashable, because it sits in the static
treedef of its owning ``ScalarField``.

The record stores the **physical** unit and a ``nondimensional`` flag;
the reported :attr:`FieldMetadata.units` is derived from the pair
(``design/research/units_metadata_investigation.md``, owner rulings
2026-08-12). One stored truth, so a nondimensional assembly cannot
leave a stale physical claim behind, and the physical unit survives
for the ``dimensional_factor`` stamp and for conversion back.
"""
# Wave 2: FieldMetadata
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

#: "nobody declared a unit" — never a CF claim. The iris/cf-units
#: spelling: ``Unit("unknown").is_unknown()`` is True while
#: ``is_dimensionless()`` is False, which is exactly the intent. The
#: former ``"n/a"`` was udunits-unparseable, and omitting the
#: attribute would assert dimensionless (CF 3.1) — a claim an
#: unannotated field has not earned.
UNKNOWN_UNITS = "unknown"

#: the CF spelling of a dimensionless quantity (canonical; ``""`` is
#: udunits' *unknown* sentinel, not dimensionless)
DIMENSIONLESS_UNITS = "1"


#: the identity slots — everything the record says about *which*
#: quantity this is. ``nondimensional`` is deliberately absent: it
#: describes the value system, not the quantity (see
#: :meth:`FieldMetadata.cleared`).
_IDENTITY_SLOTS: tuple[str, ...] = (
    "name", "long_name", "physical_units", "nc_attrs")


def _normalize_nc_attrs(
    nc_attrs: Mapping[str, str] | tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[str, str], ...]:
    """Normalize nc-attrs to the canonical sorted tuple of pairs."""
    if nc_attrs is None:
        return ()
    if isinstance(nc_attrs, tuple):
        return tuple(sorted(nc_attrs))
    return tuple(sorted(nc_attrs.items()))


@dataclass(frozen=True)
class FieldMetadata:

    """
    Immutable, hashable annotation for a ScalarField.

    Description
    -----------
    Name/units/nc-attrs only — discretization content (position, BC
    types, spectral flags) lives in the function space. ``nc_attrs``
    is canonically a sorted tuple of pairs so the dataclass stays
    hashable; use :meth:`create` to build one from a mapping.

    The unit is stored as the **physical** string plus the
    :attr:`nondimensional` flag the assembly stamps from the model's
    scaling; :attr:`units` renders the pair. ``units=`` is accepted
    as the writing sugar for ``physical_units=`` throughout
    (:meth:`create`, :meth:`replace`), so declaration sites spell
    the physical unit and never the scaling.

    Parameters
    ----------
    name : str
        Short variable name (default: "unnamed").
    long_name : str
        Descriptive nc-style name (default: "Unnamed").
    physical_units : str
        The unit the quantity carries on a dimensional model
        (default: :data:`UNKNOWN_UNITS`).
    nondimensional : bool
        Whether the stored values are nondimensionalized (default:
        False).
    nc_attrs : tuple[tuple[str, str], ...]
        Extra netCDF attributes as sorted (key, value) pairs
        (default: ()).
    """

    name: str = "unnamed"
    long_name: str = "Unnamed"
    physical_units: str = UNKNOWN_UNITS
    nondimensional: bool = False
    nc_attrs: tuple[tuple[str, str], ...] = ()

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def units(self) -> str:
        """
        The reported unit string (the CF claim).

        Description
        -----------
        :data:`DIMENSIONLESS_UNITS` on a nondimensional model, the
        physical unit otherwise. An **undeclared** unit stays
        :data:`UNKNOWN_UNITS` in both variants: nondimensionalizing
        an unknown quantity yields an unknown quantity, not a
        dimensionless one.
        """
        if self.physical_units == UNKNOWN_UNITS:
            return UNKNOWN_UNITS
        if self.nondimensional:
            return DIMENSIONLESS_UNITS
        return self.physical_units

    @property
    def units_declared(self) -> bool:
        """Whether a unit was declared (not the unknown sentinel)."""
        return self.physical_units != UNKNOWN_UNITS

    # ================================================================
    #  Construction
    # ================================================================
    @classmethod
    def create(
        cls,
        name: str = "unnamed",
        long_name: str = "Unnamed",
        units: str = UNKNOWN_UNITS,
        nc_attrs: Mapping[str, str] | None = None,
        *,
        nondimensional: bool = False,
    ) -> FieldMetadata:
        """
        Build a record, accepting a mapping for nc_attrs.

        Parameters
        ----------
        name : str, optional
            Short variable name (default: "unnamed").
        long_name : str, optional
            Descriptive nc-style name (default: "Unnamed").
        units : str, optional
            The **physical** units annotation (default:
            :data:`UNKNOWN_UNITS`).
        nc_attrs : Mapping[str, str] | None, optional
            Extra netCDF attributes; normalized to the sorted tuple
            of pairs (default: None).
        nondimensional : bool, optional
            Whether the stored values are nondimensionalized
            (default: False).

        Returns
        -------
        FieldMetadata
            The normalized metadata record.
        """
        return cls(name=name, long_name=long_name,
                   physical_units=units,
                   nondimensional=nondimensional,
                   nc_attrs=_normalize_nc_attrs(nc_attrs))

    def cleared(self) -> FieldMetadata:
        """
        Drop the identity; keep the scaling frame.

        Description
        -----------
        What a **dimension-changing** operation leaves behind
        (``f * g``, ``f / g``, ``f ** n``, ``diff``, the reductions).
        The name, long name, unit and nc-attrs of the operands say
        nothing about the result — ``u * b`` is neither a velocity
        nor a buoyancy — so those ops reset them; the
        unit-preserving ops keep what they can instead
        (:meth:`merged`, and the ``ScalarField`` class doc).
        :attr:`nondimensional`
        is different in kind: it describes the **value system** the
        numbers live in, not the quantity they measure, and anything
        computed from nondimensionalized values is itself
        nondimensionalized. So it rides through the algebra, and a
        derived quantity that re-declares its annotation on an
        algebra result renders correctly without its declaration
        site having to restate the scaling.

        Returns
        -------
        FieldMetadata
            A default record carrying only the scaling frame.
        """
        return FieldMetadata(nondimensional=self.nondimensional)

    def merged(self, other: FieldMetadata) -> FieldMetadata:
        """
        Keep what both operands agree on; drop the rest.

        Description
        -----------
        What a **unit-preserving binary** operation (``f + g``,
        ``f - g``) leaves behind. Addition is only meaningful
        between like quantities, so an annotation both operands
        state is the annotation of the result — while a *conflict*
        has no correct answer and falls back to the default rather
        than picking a side. Applied per slot (xarray's
        ``combine_attrs="drop_conflicts"`` rule), so ``u + u_bar``
        with matching ``units="m/s"`` but differing names reports
        the unit and no name. :attr:`nondimensional` is not an
        identity slot and always comes from ``self``, exactly as in
        :meth:`cleared`.

        The record is returned **identically** (``is``) when the two
        agree in full, which keeps the componentwise re-attachment
        in ``VectorField`` on its cheap path.

        Parameters
        ----------
        other : FieldMetadata
            The right operand's record.

        Returns
        -------
        FieldMetadata
            The agreed annotation over the scaling frame of
            ``self``.
        """
        if self == other:
            return self
        agreed = {
            slot: getattr(self, slot) for slot in _IDENTITY_SLOTS
            if getattr(self, slot) == getattr(other, slot)}
        return FieldMetadata(nondimensional=self.nondimensional,
                             **agreed)

    def replace(self, **changes: object) -> FieldMetadata:
        """
        Functional update (dataclasses.replace wrapper).

        Parameters
        ----------
        **changes : object
            Field values to replace; ``units`` is accepted as sugar
            for ``physical_units`` and ``nc_attrs`` may be given as
            a mapping.

        Returns
        -------
        FieldMetadata
            The updated record; ``self`` is unchanged.
        """
        if "nc_attrs" in changes:
            changes["nc_attrs"] = _normalize_nc_attrs(
                changes["nc_attrs"])
        if "units" in changes:
            if "physical_units" in changes:
                raise ValueError(
                    "units= is sugar for physical_units=; pass one")
            changes["physical_units"] = changes.pop("units")
        return dataclasses.replace(self, **changes)
