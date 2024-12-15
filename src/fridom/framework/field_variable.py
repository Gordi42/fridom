"""field_variable.py - FieldVariable class for the fridom framework."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

import fridom.framework as fr
from fridom.framework.grid.fft_padding import FFTPadding

if TYPE_CHECKING:
    import xarray as xr

@partial(fr.utils.jaxify, dynamic=("position", ))
@dataclass
class FieldMetadata:

    """
    Metadata for the FieldVariable.

    Description
    -----------
    The FieldMetadata class contains all metadata for the FieldVariable. This
    includes the name, long name, units, and additional attributes for the
    NetCDF file or xarray. The metadata also contains information about the
    position of the FieldVariable on the grid, the topology, and the boundary
    conditions.

    Parameters
    ----------
    name : str (default "unnamed")
        Name of the FieldVariable
    long_name : str (default "Unnamed")
        Long name of the FieldVariable
    units : str (default "n/a")
        Units of the FieldVariable
    nc_attrs : dict | None (default None)
        Additional attributes for the NetCDF file or xarray
    is_spectral : bool (default False)
        True if the FieldVariable should be initialized in spectral space
    topo : list[bool] | None (default None)
        Topology of the FieldVariable. If None, the FieldVariable is
        assumed to be fully extended in all directions. If a list of booleans
        is given, the FieldVariable has no extend in the directions where the
        corresponding entry is False.
    position : fr.grid.Position | None (default None)
        Position of the FieldVariable on the grid
    bc_types : tuple[BCType] | None (default None)
        Tuple of BCType objects that specify the type of boundary condition
        in each direction. If None, the default boundary conditions is Neumann.
    _flags : dict (default {"NO_ADV": False,
                            "ENABLE_MIXING": False,
                            "ENABLE_FRICTION": False})
        Dictionary with flag options for the FieldVariable

    """

    name: str = "unnamed"
    long_name: str = "Unnamed"
    units: str = "n/a"
    nc_attrs: dict | None = None
    is_spectral: bool = False
    topo: list[bool] | None = None
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
            self.topo = [True] * mset.grid.n_dims

        if self.bc_types is None:
            self.bc_types = [fr.grid.BCType.NEUMANN] * mset.grid.n_dims

    def to_serializable(self) -> dict:
        """Convert the FieldMetadata to a serializable dictionary."""
        res = copy(self.__dict__)
        res["nc_attrs"] = [str(key) for key in self.nc_attrs]
        res.update(self.nc_attrs)
        res["is_spectral"] = int(self.is_spectral)
        res["topo"] = tuple(int(x) for x in self.topo)
        res["_bc_types"] = [x.value for x in self.bc_types]
        res["position"] = [x.value for x in self.position.positions]
        res["_flags"] = [str(key) for key in self._flags]
        for flag, value in self._flags.items():
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
                   topo=tuple(bool(x) for x in data["topo"]),
                   position=fr.grid.Position(
                       [fr.grid.AxisPosition(x) for x in data["position"]]),
                   _bc_types=tuple(fr.grid.BCType(x) for x in data["_bc_types"]),
                   _flags={key: bool(data[key]) for key in data["_flags"]})

    @property
    def bc_types(self) -> tuple[fr.grid.BCType] | None:
        """The boundary condition types for the FieldVariable."""
        return self._bc_types

    @bc_types.setter
    def bc_types(self, bc_types: tuple[fr.grid.BCType] | None) -> None:
        if bc_types is not None and len(bc_types) != len(self.position.positions):
            msg = "Number of BCType objects must match the number of positions"
            raise ValueError(msg)
        self._bc_types = tuple(bc_types)

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the FieldVariable."""
        return self._flags

    @flags.setter
    def flags(self, update: dict) -> None:
        for key, value in update.items():
            if key not in self._flags:
                msg = f"Flag {key} not available. "
                msg += f"Available flags: {self._flags}"
                raise KeyError(msg)
            if not isinstance(value, bool):
                fr.log.warning(f"Flag {key} must be a boolean")
                msg = f"Flag {key} is of type {type(value)}"
                raise TypeError(msg)
            self._flags[key] = value


@partial(fr.utils.jaxify, dynamic=("_arr", "_mdata"))
class FieldVariable:

    """
    Class for field variables in the framework.

    Description
    -----------
    TODO

    Parameters
    ----------
    mset : ModelSettings
        ModelSettings object
    mdata : FieldMetadata | None (default None)
        Metadata for the FieldVariable
    arr : ndarray (default None)
        The array to be wrapped

    """

    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 mdata: FieldMetadata | None = None,
                 arr: ndarray | None = None,
                 **kwargs: any) -> None:
        # create new metadata object if not provided
        mdata = mdata or FieldMetadata()

        # set the attributes from the kwargs
        for key, value in kwargs.items():
            setattr(mdata, key, value)

        # set default values of the metadata
        mdata.set_default(mset)

        # The underlying array
        if arr is None:
            data = mset.grid.create_array(
                pad=True,
                spectral=mdata.is_spectral,
                topo=tuple(mdata.topo))
        else:
            conf = fr.config
            dtype = conf.dtype_comp if mdata.is_spectral else conf.dtype_real
            data = conf.ncp.array(arr, dtype=dtype)

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self._mdata = mdata
        self._arr = data
        self._mset = mset

    def fft(self,
            padding: FFTPadding = FFTPadding.NOPADDING) -> FieldVariable:
        """
        Forward Fourier transform the FieldVariable.

        Parameters
        ----------
        padding : FFTPadding
            Zero padding option

        Returns
        -------
        FieldVariable
            The FieldVariable in spectral space.

        """
        if not self.grid.fourier_transform_available:
            msg = "Fourier transform not available for this grid"
            raise NotImplementedError(msg)

        if self.is_spectral:
            msg = "FieldVariable is in spectral space, cannot perform fft"
            raise ValueError(msg)

        transformed_arr = self.grid.fft(
            arr=self.arr,
            padding=padding,
            bc_types=self.bc_types,
            positions=self.position.positions)

        conf = fr.config
        transformed_arr = conf.ncp.array(transformed_arr, dtype=conf.dtype_comp)

        return FieldVariable(self.mset,
                             arr=transformed_arr,
                             mdata=deepcopy(self.mdata),
                             is_spectral=True)

    def ifft(self,
             padding: FFTPadding = FFTPadding.NOPADDING) -> FieldVariable:
        """
        Inverse Fourier transform of the FieldVariable.

        Parameters
        ----------
        padding : FFTPadding
            Zero padding option

        Returns
        -------
        FieldVariable
            The FieldVariable in physical space.

        """
        if not self.grid.fourier_transform_available:
            msg = "Fourier transform not available for this grid"
            raise NotImplementedError(msg)

        if not self.is_spectral:
            msg = "FieldVariable is not in spectral space, cannot perform fft"
            raise ValueError(msg)

        transformed_arr = self.grid.ifft(
            arr=self.arr,
            padding=padding,
            bc_types=self.bc_types,
            positions=self.position.positions)

        # only keep the real part
        conf = fr.config
        transformed_arr = conf.ncp.array(transformed_arr.real, dtype=conf.dtype_real)

        return FieldVariable(self.mset,
                             arr=transformed_arr,
                             mdata=deepcopy(self.mdata),
                             is_spectral=False)

    def sync(self) -> FieldVariable:
        """Synchronize the FieldVariable (exchange boundary values)."""
        if self.is_spectral:
            # nothing to synchronize in spectral space
            return self
        self.arr = self.grid.sync(self.arr)
        self.apply_water_mask()
        return self

    def unpad(self) -> ndarray:
        """Remove padding from the FieldVariable."""
        if self.is_spectral:
            msg = "FieldVariable is in spectral space, cannot unpad"
            raise ValueError(msg)
        return self.grid.unpad(self.arr)

    def apply_water_mask(self) -> FieldVariable:
        """Apply boundary conditions to the FieldVariable."""
        if self.is_spectral:
            msg = "FieldVariable is in spectral space, cannot apply water mask"
            raise ValueError(msg)
        self.arr *= self.grid.water_mask.get_mask(self.position)
        return self

    def get_mesh(self) -> tuple[ndarray]:
        """Get the meshgrid of the FieldVariable."""
        return self.grid.get_mesh(self.position, self.is_spectral)

    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self, axis: int, order: int = 1) -> FieldVariable:
        r"""
        Compute the partial derivative along an axis.

        .. math::
            \partial_i^n f

        with axis :math:`i` and order :math:`n`.

        Parameters
        ----------
        axis : int
            The axis along which to differentiate.
        order : int
            The order of the derivative. Default is 1.

        Returns
        -------
        FieldVariable
            The derivative of the field along the specified axis.

        """
        return self.grid.diff_module.diff(self, axis, order)

    def grad(self, axes: list[int] | None = None ) -> tuple[FieldVariable | None]:
        r"""
        Compute the gradient.

        .. math::
            \nabla f =
            \begin{pmatrix} \partial_1 f \\ \dots \\ \partial_n f \end{pmatrix}

        Parameters
        ----------
        axes : list[int] | None (default is None)
            The axes along which to compute the gradient. If `None`, the
            gradient is computed along all axes.

        Returns
        -------
        tuple[FieldVariable | None]
            The gradient of the field along the specified axes. The list contains
            the gradient components along each axis. Axis which are not included
            in `axes` will have a value of `None`.
            E.g. for a 3D grid, `diff.grad(f, axes=[0, 2])` will return
            `[df/dx, None, df/dz]`.

        """
        return self.grid.diff_module.grad(self, axes)

    def laplacian(self,
                  axes: tuple[int] | None = None,
                  ) -> FieldVariable:
        r"""
        Compute the Laplacian.

        .. math::
            \nabla^2 f = \sum_{i=1}^n \partial_i^2 f

        Parameters
        ----------
        axes : tuple[int] | None (default is None)
            The axes along which to compute the Laplacian. If `None`, the
            Laplacian is computed along all axes.

        Returns
        -------
        FieldVariable
            The Laplacian of the field.

        """
        return self.grid.diff_module.laplacian(self, axes)

    def interpolate(self, destination: fr.grid.Position) -> FieldVariable:
        """
        Interpolate the field to the destination position.

        Parameters
        ----------
        destination : fr.grid.Position
            The position to interpolate to.

        Returns
        -------
        `FieldVariable`
            The interpolated field.

        """
        return self.grid.interp_module.interpolate(self, destination)

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key: slice | tuple[slice]) -> ndarray:
        return self.arr[key]

    def __setitem__(self, key: slice | tuple[slice], value: ndarray | float) -> None:
        new_arr = fr.utils.modify_array(self.arr, key, value)
        self.arr = new_arr

    # ================================================================
    #  Pickling
    # ================================================================

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __copy__(self) -> FieldVariable:
        # copy the array and the metadata but not the model settings
        arr = deepcopy(self.arr)
        mdata = deepcopy(self.mdata)
        return FieldVariable(mset=self.mset, mdata=mdata, arr=arr)

    # ==================================================================
    #  Display methods
    # ==================================================================

    @property
    def info(self) -> dict:
        """Dictionary with information about the field."""
        res = {}
        res["name"] = self.name
        res["long_name"] = self.long_name
        res["units"] = self.units
        res["is_spectral"] = self.is_spectral
        res["position"] = self.position
        res["topo"] = self.topo
        res["bc_types"] = self.bc_types
        enabled_flags = [key for key, value in self.flags.items() if value]
        res["enabled_flags"] = enabled_flags
        return res

    def __repr__(self) -> str:
        res = "FieldVariable"
        for key, value in self.info.items():
            res += f"\n  - {key}: {value}"
        return res

    # ================================================================
    #  xarray conversion
    # ================================================================

    @property
    def xr(self) -> xr.DataArray:
        """Convert to xarray DataArray."""
        return self.xrs[:]

    @property
    def xrs(self) -> fr.utils.SliceableAttribute[xr.DataArray]:
        """
        Convert a slice of the FieldVariable to xarray DataArray.

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The slice to apply to the FieldVariable.

        Example
        -------
        Let `f` be a large 3D FieldVariable and we want to convert the top
        of the field to an xarray DataArray. To avoid loading the whole field
        into memory, we can use slicing:

        .. code-block:: python

            data_array = f.xrs[:,:,-1]  # Only the top of the field

        """
        def slicer(key: int | slice | tuple[int | slice]) -> xr.DataArray:
            import xarray as xr
            fv = self
            # convert key to tuple
            ndim = fv.grid.n_dims
            key = [key] if not isinstance(key, (tuple, list)) else list(key)
            key += [slice(None)] * (ndim - len(key))

            for i in range(ndim):
                # set non-extended axes to 0
                if not fv.topo[i]:
                    key[i] = slice(0,1)
                if isinstance(key[i], int):
                    if key[i] < 0:
                        key[i] = slice(key[i]-1, key[i])
                    else:
                        key[i] = slice(key[i], key[i]+1)

            arr = fv.grid.domain_decomp.gather(
                fv.arr, tuple(key), spectral=fv.is_spectral)

            # get the coordinates
            if ndim <= 3:
                if fv.is_spectral:
                    all_dims = tuple(["kx", "ky", "kz"][:ndim])
                else:
                    all_dims = tuple(["x", "y", "z"][:ndim])
            elif fv.is_spectral:
                all_dims = tuple(f"k{i}" for i in range(ndim))
            else:
                all_dims = tuple(f"x{i}" for i in range(ndim))

            dims = []
            coords = {}
            for axis in range(fv.grid.n_dims):
                if arr.shape[axis] == 1:
                    # skip non-extended axes
                    continue

                dim = all_dims[axis]
                dims.append(dim)
                if fv.is_spectral:
                    x_sel = fv.grid.k_global[axis][key[axis]]
                else:
                    x_sel = fv.grid.x_global[axis][key[axis]]
                coords[dim] = fr.utils.to_numpy(x_sel)

            # reverse the dimensions
            dims.reverse()

            all_attrs = fv.mdata.to_serializable()

            dv = xr.DataArray(
                fr.utils.to_numpy(np.squeeze(arr).T),
                coords=coords,
                dims=tuple(dims),
                name=fv.name,
                attrs=all_attrs)

            x_unit = "1/m" if fv.is_spectral else "m"
            for dim in dims:
                dv[dim].attrs["units"] = x_unit
            return dv
        return fr.utils.SliceableAttribute(slicer)

    @classmethod
    def from_xarray(cls, mset: fr.ModelSettingsBase, da: xr.DataArray) -> FieldVariable:
        """
        Create a FieldVariable from an xarray DataArray.

        Parameters
        ----------
        mset : ModelSettingsBase
            The model settings object.
        da : xr.DataArray
            The xarray DataArray to convert.

        Returns
        -------
        FieldVariable
            The FieldVariable.

        """
        conf = fr.config
        # load the metadata
        mdata = FieldMetadata.from_serializable(da.attrs)
        # convert the array to backend
        arr = da.to_numpy().T
        if mdata.is_spectral:
            arr_real = conf.ncp.array(arr["r"])
            arr_imag = conf.ncp.array(arr["i"])
            arr = conf.ncp.array(arr_real + 1j * arr_imag, dtype=conf.dtype_comp)
        else:
            arr = conf.ncp.array(arr, dtype=conf.dtype_real)
            # pad the array
            arr = mset.grid.pad(arr)
        # create the FieldVariable
        field = cls(mset=mset, mdata=mdata, arr=arr)
        # synchronize the field
        return field.sync()

    def to_netcdf(self, path: str) -> None:
        """
        Save the FieldVariable to a NetCDF file.

        Parameters
        ----------
        path : str
            The path to the NetCDF file.

        """
        self.xr.to_netcdf(path, auto_complex=True)

    @classmethod
    def from_netcdf(cls, mset: fr.ModelSettingsBase, path: str) -> FieldVariable:
        """
        Create a FieldVariable from a NetCDF file.

        Parameters
        ----------
        mset : ModelSettingsBase
            The model settings object.
        path : str
            The path to the NetCDF file.

        Returns
        -------
        FieldVariable
            The FieldVariable.

        """
        import xarray as xr
        da = xr.open_dataarray(path)
        return cls.from_xarray(mset, da)

    # ==================================================================
    #  OTHER METHODS
    # ==================================================================

    def has_nan(self) -> bool:
        """Check if the FieldVariable contains NaN values."""
        ncp = fr.config.ncp
        return ncp.any(ncp.isnan(self.arr))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def arr(self) -> ndarray:
        """The underlying array."""
        return self._arr

    @arr.setter
    def arr(self, arr: ndarray) -> None:
        self._arr = arr

    @property
    def mdata(self) -> FieldMetadata:
        """The metadata of the FieldVariable."""
        return self._mdata

    @mdata.setter
    def mdata(self, mdata: FieldMetadata) -> None:
        self._mdata = mdata

    @property
    def name(self) -> str:
        """The name of the FieldVariable."""
        return self.mdata.name

    @name.setter
    def name(self, name: str) -> None:
        self.mdata.name = name

    @property
    def long_name(self) -> str:
        """The long name of the FieldVariable."""
        return self.mdata.long_name

    @long_name.setter
    def long_name(self, long_name: str) -> None:
        self.mdata.long_name = long_name

    @property
    def units(self) -> str:
        """The unit of the FieldVariable."""
        return self.mdata.units

    @units.setter
    def units(self, units: str) -> None:
        self.mdata.units = units

    @property
    def nc_attrs(self) -> dict:
        """Dictionary with additional attributes for the NetCDF file or xarray."""
        return self.mdata.nc_attrs

    @nc_attrs.setter
    def nc_attrs(self, nc_attrs: dict) -> None:
        self.mdata.nc_attrs = nc_attrs

    @property
    def is_spectral(self) -> bool:
        """True if the FieldVariable is in spectral space."""
        return self.mdata.is_spectral

    @is_spectral.setter
    def is_spectral(self, is_spectral: bool) -> None:
        self.mdata.is_spectral = is_spectral

    @property
    def topo(self) -> list[bool]:
        """
        Topology of the FieldVariable.

        Description
        -----------
        Field Variables do not have to be extended in all directions. For
        example, one might want to create a 2D forcing field for a 3D simulation,
        that only depends on x and y. In this case, the topo of the FieldVariable
        would be [True, True, False].
        """
        return self.mdata.topo

    @property
    def position(self) -> fr.grid.Position:
        """The position of the FieldVariable on the staggered grid."""
        return self.mdata.position

    @position.setter
    def position(self, position: fr.grid.Position) -> None:
        self.mdata.position = position

    @property
    def bc_types(self) -> tuple[fr.grid.BCType] | None:
        """The boundary condition types for the FieldVariable."""
        return self.mdata.bc_types

    @bc_types.setter
    def bc_types(self, bc_types: tuple[fr.grid.BCType] | None) -> None:
        self.mdata.bc_types = bc_types

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the FieldVariable."""
        return self.mdata.flags

    @flags.setter
    def flags(self, flags: dict) -> None:
        self.mdata.flags = flags

    @property
    def mset(self) -> fr.ModelSettingsBase:
        """The model settings object."""
        return self._mset

    @property
    def grid(self) -> fr.grid.GridBase:
        """The grid object."""
        return self._mset.grid

    # ==================================================================
    #  ARITHMETIC OPERATIONS
    # ==================================================================

    def abs(self) -> FieldVariable:
        """Absolute values of the FieldVariable."""
        arr = fr.config.ncp.abs(self.arr)
        return FieldVariable(mset=self.mset, mdata=deepcopy(self.mdata), arr=arr)

    def __abs__(self) -> FieldVariable:
        return self.abs()

    def sum(self, axes: tuple[int] | None = None) -> float:
        """Sum of the FieldVariable over the whole domain in the specified axes."""
        domain = self.grid.domain_decomp
        return domain.sum(self.arr, axes=axes, spectral=self.is_spectral)

    def __sum__(self) -> float:
        return self.sum()

    def max(self, axes: tuple[int] | None = None) -> float:
        """Maximum value of the FieldVariable over the whole domain."""
        domain = self.grid.domain_decomp
        return domain.max(self.arr, axes=axes, spectral=self.is_spectral)

    def __max__(self) -> float:
        return self.max()

    def min(self, axes: tuple[int] | None = None) -> float:
        """Minimum value of the FieldVariable over the whole domain."""
        domain = self.grid.domain_decomp
        return domain.min(self.arr, axes=axes, spectral=self.is_spectral)

    def __min__(self) -> float:
        return self.min()

    def integrate(self) -> float:
        """Global integral of the FieldVariable."""
        if self.is_spectral:
            msg = "Integration not available for spectral fields"
            raise NotImplementedError(msg)
        domain = self.grid.domain_decomp
        return domain.sum(self.arr * self.grid.dV)

    def norm_l2(self) -> float:
        """Compute the numpy.linalg.norm of the FieldVariable."""
        norm = fr.config.ncp.linalg.norm(self.unpad())**2
        return fr.config.ncp.sqrt(norm)

    @staticmethod
    def _apply_operation(
        op: callable, field: FieldVariable, other: any) -> FieldVariable:
        new_mdata = deepcopy(field.mdata)
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(field.topo, other.topo)]
            new_mdata.topo = topo
            result = op(field.arr, other.arr)
        else:
            result = op(field.arr, other)

        return FieldVariable(mset=field.mset, mdata=new_mdata, arr=result)

    def __add__(self, other: any) -> FieldVariable:
        return self._apply_operation(lambda x, y: x + y, self, other)

    def __radd__(self, other: any) -> FieldVariable:
        return self.__add__(other)

    def __sub__(self, other: any) -> FieldVariable:
        return self._apply_operation(lambda x, y: x - y, self, other)

    def __rsub__(self, other: any) -> FieldVariable:
        return self._apply_operation(lambda x, y: y - x, self, other)

    def __mul__(self, other: any) -> FieldVariable:
        return self._apply_operation(lambda x, y: x * y, self, other)

    def __rmul__(self, other: any) -> FieldVariable:
        return self.__mul__(other)

    def __truediv__(self, other: any) -> FieldVariable:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: x / y, self, other)

    def __rtruediv__(self, other: any) -> FieldVariable:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: y / x, self, other)

    def __pow__(self, other: any) -> FieldVariable:
        return self._apply_operation(lambda x, y: x ** y, self, other)

    def __neg__(self) -> FieldVariable:
        """Negate the FieldVariable."""
        return FieldVariable(mset=self.mset,
                             mdata=deepcopy(self.mdata),
                             arr=-self.arr)
