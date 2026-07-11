"""dataclass for the metadata of a ScalarField."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("position", ))
@dataclass
class FieldMetadata:

    """
    Metadata for the ScalarField.

    Description
    -----------
    The FieldMetadata class contains all metadata for the ScalarField. This
    includes the name, long name, units, and additional attributes for the
    NetCDF file or xarray. The metadata also contains information about the
    position of the ScalarField on the grid, the topology, and the boundary
    conditions.

    Parameters
    ----------
    name : str (default "unnamed")
        Name of the ScalarField
    long_name : str (default "Unnamed")
        Long name of the ScalarField
    units : str (default "n/a")
        Units of the ScalarField
    nc_attrs : dict | None (default None)
        Additional attributes for the NetCDF file or xarray
    is_spectral : bool (default False)
        True if the ScalarField should be initialized in spectral space
    topo : list[bool] | None (default None)
        Topology of the ScalarField. If None, the ScalarField is
        assumed to be fully extended in all directions. If a list of booleans
        is given, the ScalarField has no extend in the directions where the
        corresponding entry is False.
    position : fr.grid.Position | None (default None)
        Position of the ScalarField on the grid
    bc_types : tuple[BCType] | None (default None)
        Tuple of BCType objects that specify the type of boundary condition
        in each direction. If None, the default boundary conditions is Neumann.
    _flags : dict (default {"NO_ADV": False,
                            "ENABLE_MIXING": False,
                            "ENABLE_FRICTION": False})
        Dictionary with flag options for the ScalarField

    """

    name: str = "unnamed"
    long_name: str = "Unnamed"
    units: str = "n/a"
    nc_attrs: dict | None = None
    is_spectral: bool = False
    _topo: tuple[bool] | None = None
    position: fr.grid.Position | None = None
    _bc_types: tuple[fr.grid.BCType] | None = None
    _flags: dict = field(
        default_factory=lambda: {"NO_ADV": False,
                                 "ENABLE_MIXING": False,
                                 "ENABLE_FRICTION": False})

    def set_default(self, mset: fr.ModelSettingsBase) -> None:
        """Set None values to default values."""
        if self.nc_attrs is None:
            self.nc_attrs = {}

        if self.position is None:
            self.position = mset.grid.cell_center

        if self.topo is None:
            self.topo = tuple([True] * mset.grid.n_dims)

        if self.bc_types is None:
            self.bc_types = [fr.grid.BCType.NEUMANN] * mset.grid.n_dims

    def __copy__(self) -> FieldMetadata:
        return deepcopy(self)

    def copy(self) -> FieldMetadata:
        """Create a deep copy of the FieldMetadata."""
        return deepcopy(self)

    # ================================================================
    #  Serialization
    # ================================================================

    def to_serializable(self) -> dict:
        """Convert the FieldMetadata to a serializable dictionary."""
        res = {
            "name": self.name,
            "long_name": self.long_name,
            "units": self.units,
            "nc_attrs": [str(key) for key in self.nc_attrs],
            "is_spectral": int(self.is_spectral),
            "topo": tuple(int(x) for x in self.topo),
            "position": [x.value for x in self.position.positions],
            "bc_types": [x.value for x in self.bc_types],
            "flags": [str(key) for key in self.flags],
        }
        res.update(self.nc_attrs)
        for flag, value in self.flags.items():
            res[flag] = int(value)
        return res

    @classmethod
    def from_serializable(cls, data: dict) -> FieldMetadata:
        """Create a FieldMetadata object from a serializable dictionary."""
        return cls(name=data["name"],
                   long_name=data["long_name"],
                   units=data["units"],
                   nc_attrs={key: data[key] for key in data["nc_attrs"]},
                   is_spectral=bool(data["is_spectral"]),
                   _topo=tuple(bool(x) for x in data["topo"]),
                   position=fr.grid.Position(
                       [fr.grid.AxisPosition(x) for x in data["position"]]),
                   _bc_types=tuple(
                       fr.grid.BCType(x) for x in data["bc_types"]),
                   _flags={key: bool(data[key]) for key in data["flags"]})

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def topo(self) -> tuple[bool]:
        """
        The topology of the Scalar Field.

        Description
        -----------
        The topology of a scalar field determines whether the field lives in
        a certain direction. If the topology is True in a direction, the field
        is fully extended in this direction, if it is False, the field has no
        extend in this direction.
        """
        return self._topo

    @topo.setter
    def topo(self, topo: tuple[bool] | list[bool]) -> None:
        self._topo = tuple(topo)

    @property
    def bc_types(self) -> tuple[fr.grid.BCType] | None:
        """The boundary condition types for the ScalarField."""
        return self._bc_types

    @bc_types.setter
    def bc_types(self, bc_types: tuple[fr.grid.BCType] | None) -> None:
        if (bc_types is not None
                and len(bc_types) != len(self.position.positions)):
            msg = "Number of BCType objects must match the number of positions"
            raise ValueError(msg)
        self._bc_types = tuple(bc_types)

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the ScalarField."""
        return self._flags

    @flags.setter
    def flags(self, update: dict) -> None:
        for key, value in update.items():
            if key not in self._flags:
                msg = f"Flag {key} not available. "
                msg += f"Available flags: {self._flags}"
                raise KeyError(msg)
            if not isinstance(value, bool):
                msg = f"Flag {key} must be a boolean, "
                msg += f"but is of type {type(value)}"
                raise TypeError(msg)
            self._flags[key] = value
