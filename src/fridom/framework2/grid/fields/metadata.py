"""
``FieldMetadata``: annotation metadata for fields.

Description
-----------
Owning class doc: ``design/specs/grid/classes/fields.md``. Pure
annotation for naming and I/O (name, units, nc-attrs) with no
discretization content: metadata never influences dispatch, dtype,
shape, or algebra. Frozen and hashable, because it sits in the static
treedef of its owning ``ScalarField``.
"""
# Wave 2: FieldMetadata
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


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

    Parameters
    ----------
    name : str
        Short variable name (default: "unnamed").
    long_name : str
        Descriptive nc-style name (default: "Unnamed").
    units : str
        Physical units annotation (default: "n/a").
    nc_attrs : tuple[tuple[str, str], ...]
        Extra netCDF attributes as sorted (key, value) pairs
        (default: ()).
    """

    name: str = "unnamed"
    long_name: str = "Unnamed"
    units: str = "n/a"
    nc_attrs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        name: str = "unnamed",
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
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
            Physical units annotation (default: "n/a").
        nc_attrs : Mapping[str, str] | None, optional
            Extra netCDF attributes; normalized to the sorted tuple
            of pairs (default: None).

        Returns
        -------
        FieldMetadata
            The normalized metadata record.
        """
        return cls(name=name, long_name=long_name, units=units,
                   nc_attrs=_normalize_nc_attrs(nc_attrs))

    def replace(self, **changes: object) -> FieldMetadata:
        """
        Functional update (dataclasses.replace wrapper).

        Parameters
        ----------
        **changes : object
            Field values to replace; ``nc_attrs`` may be given as a
            mapping and is normalized.

        Returns
        -------
        FieldMetadata
            The updated record; ``self`` is unchanged.
        """
        if "nc_attrs" in changes:
            changes["nc_attrs"] = _normalize_nc_attrs(
                changes["nc_attrs"])
        return dataclasses.replace(self, **changes)
