"""Scalar field class definition."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
    from numpy import ndarray

@partial(fr.utils.jaxify, dynamic=("_arr", "_mdata"))
class ScalarField(fr.FieldBase):

    """
    A scalar mapping from grid space to real / complex numbers.

    Description
    -----------
    A scalar field is the most basic field in FRIDOM. It is a mapping from the
    grid space to real or complex numbers. It is used to represent scalar
    quantities like pressure, temperature, etc. Essentially, a scalar field is
    wrapper around a numpy-like array with additional metadata and methods.

    Parameters
    ----------
    mset : fr.ModelSettingsBase
        The model settings object.
    mdata : fr.FieldMetadata, optional
        The metadata object for the field. If not provided, a new one is created.
    arr : ndarray, optional
        The underlying array of the field. If not provided, a new array is created.

    """

    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 mdata: fr.FieldMetadata | None = None,
                 arr: ndarray | None = None,
                 **kwargs: any) -> None:
        super().__init__(mset=mset)
        # create new metadata object if not provided
        mdata = mdata or fr.FieldMetadata()

        # set default values of the metadata
        mdata.set_default(mset)

        # set the attributes from the kwargs
        for key, value in kwargs.items():
            setattr(mdata, key, value)

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

    # ================================================================
    #  General methods
    # ================================================================

    def fft(self,  # noqa: D102
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            ) -> ScalarField:
        self._fft_possible()
        if not all(self.topo):
            msg = "Cannot transform non full domain fields"
            raise NotImplementedError(msg)

        transformed_arr = self.grid.fft(
            arr=self.arr,
            padding=padding,
            bc_types=self.bc_types,
            positions=self.position.positions)

        conf = fr.config
        transformed_arr = conf.ncp.array(transformed_arr, dtype=conf.dtype_comp)

        return ScalarField(self.mset,
                           arr=transformed_arr,
                           mdata=deepcopy(self.mdata),
                           is_spectral=True)

    def ifft(self,  # noqa: D102
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             ) -> ScalarField:
        self._ifft_possible()
        if not all(self.topo):
            msg = "Cannot transform non full domain fields"
            raise NotImplementedError(msg)

        transformed_arr = self.grid.ifft(
            arr=self.arr,
            padding=padding,
            bc_types=self.bc_types,
            positions=self.position.positions)

        # only keep the real part
        conf = fr.config
        transformed_arr = conf.ncp.array(transformed_arr.real, dtype=conf.dtype_real)

        return ScalarField(self.mset,
                           arr=transformed_arr,
                           mdata=deepcopy(self.mdata),
                           is_spectral=False)

    def sync(self) -> ScalarField:  # noqa: D102
        if self.is_spectral:
            # nothing to synchronize in spectral space
            return self
        self.arr = self.grid.sync(self.arr)
        self.apply_water_mask()
        return self

    def apply_water_mask(self) -> ScalarField:  # noqa: D102
        if self.is_spectral:
            msg = "ScalarField is in spectral space, cannot apply water mask"
            raise ValueError(msg)
        self.arr *= self.grid.water_mask.get_mask(self.position)
        return self

    def has_nan(self) -> bool:  # noqa: D102
        ncp = fr.config.ncp
        return ncp.any(ncp.isnan(self.arr))

    def __copy__(self) -> ScalarField:
        # copy the array and the metadata but not the model settings
        arr = deepcopy(self.arr)
        mdata = deepcopy(self.mdata)
        return ScalarField(mset=self.mset, mdata=mdata, arr=arr)

    def unpad(self) -> ndarray:
        """
        Remove padding from the Scalar Field.

        Returns
        -------
        ndarray
            The unpadded array.

        """
        if self.is_spectral:
            msg = "Field is in spectral space, cannot unpad"
            raise ValueError(msg)
        return self.grid.unpad(self.arr)

    def get_mesh(self) -> tuple[ndarray]:
        """Get the meshgrid of the ScalarField."""
        return self.grid.get_mesh(self.position, self.is_spectral)

    def interpolate(self, destination: fr.grid.Position) -> ScalarField:
        """
        Interpolate the field to the destination position.

        Parameters
        ----------
        destination : fr.grid.Position
            The position to interpolate to.

        Returns
        -------
        ScalarField
            The interpolated field.

        """
        return self.grid.interp_module.interpolate(self, destination)

    def extend(self, topo: tuple[bool]) -> ScalarField:
        r"""
        Extend the field in the specified directions.

        Description
        -----------
        This method extends the field in the specified directions. The field
        can be extended in any direction, but it cannot be shrunk. This means
        that if the field is extended in a direction, it has to be extended in
        all directions. Values in the extended directions are copied from the
        original field, such that:

        .. math::
            f_{\text{new}}(x, y, z) = f_{\text{old}}(x, y)

        where :math:`f_{\text{new}}` is the new field extended in (x, y, z),
        and :math:`f_{\text{old}}` is the old field, extended in (x, y).

        Parameters
        ----------
        topo : tuple[bool]
            The new topology of the field.

        Returns
        -------
        ScalarField
            The extended field.

        Raises
        ------
        ValueError
            If the field is shrunk in any direction.

        """
        # check if the topology is valid (no shrinking)
        old_topo = self.topo
        for (old, new) in zip(old_topo, topo):
            if old and not new:
                msg = "Cannot shrink the field in any direction"
                raise ValueError(msg)
        # TODO(Silvano): The grid.extend method is not implemented yet
        return self.grid.extend(self, topo)

    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self, axis: int, order: int = 1) -> ScalarField:  # noqa: D102
        return self.grid.diff_module.diff(self, axis, order)

    def grad(self, axes: list[int] | None = None ) -> fr.VectorField:  # noqa: D102
        return self.grid.diff_module.grad(self, axes)

    def laplacian(self,  # noqa: D102
                  axes: tuple[int] | None = None,
                  ) -> ScalarField:
        return self.grid.diff_module.laplacian(self, axes)

    def div(self, axes: list[int] | None = None) -> None:  # noqa: D102
        _ = axes
        msg = "Divergence is not defined for scalar fields"
        raise ValueError(msg)

    # ================================================================
    #  xarray Interface
    # ================================================================

    def _convert_slice_to_xarray(self,
                                 key: int | slice | tuple[int | slice],
                                 ) -> xr.DataArray:
        import xarray as xr
        # normalize the key
        key = self._normalize_slice_key(key)

        # gather the array on the root process
        arr = self.grid.domain_decomp.gather(
            self.arr, key, spectral=self.is_spectral)

        # get the dimensions and coordinates of the slice
        dims, coords = self._get_sliced_coords(key, arr.shape)

        # reverse the dimensions
        dims.reverse()

        # get all attributes
        all_attrs = self.mdata.to_serializable()

        # create the xarray DataArray
        dv = xr.DataArray(
            fr.utils.to_numpy(np.squeeze(arr).T),
            coords=coords,
            dims=tuple(dims),
            name=self.name,
            attrs=all_attrs)

        # add the additional attributes to the coordinates
        x_unit = "1/m" if self.is_spectral else "m"
        for dim in dims:
            dv[dim].attrs["units"] = x_unit
        return dv

    def _normalize_slice_key(self,
                             key: int | slice | tuple[int | slice],
                             ) -> tuple[int | slice]:
        """Normalize the slice key to the number of dimensions."""
        ndim = self.grid.n_dims
        # convert key to list
        key = [key] if not isinstance(key, (tuple, list)) else list(key)
        # extend the key to the number of dimensions
        key += [slice(None)] * (ndim - len(key))

        for i in range(ndim):
            # set non-extended axes to 0
            if not self.topo[i]:
                key[i] = slice(0, 1)
            # convert negative indices to slices
            if isinstance(key[i], int):
                if key[i] < 0:
                    key[i] = slice(key[i]-1, key[i])
                else:
                    key[i] = slice(key[i], key[i]+1)

        return tuple(key)

    def _get_sliced_coords(self,
                           key: tuple[int | slice],
                           shape: tuple[int],
                           ) -> tuple[list, dict]:
        """Get the coordinates for a slice of the ScalarField."""
        ndim = self.grid.n_dims
        realistic_dims = 3
        # get the coordinates
        if ndim <= realistic_dims:
            dim_names = ["kx", "ky", "kz"] if self.is_spectral else ["x", "y", "z"]
            all_dims = tuple(dim_names[:ndim])
        else:
            prefix = "k" if self.is_spectral else "x"
            all_dims = tuple(f"{prefix}{i}" for i in range(ndim))

        mesh = self.grid.k_global if self.is_spectral else self.grid.x_global
        dims = []
        coords = {}
        for axis in range(self.grid.n_dims):
            if shape[axis] == 1:  # skip non-extended axes
                continue

            dim = all_dims[axis]
            dims.append(dim)
            coords[dim] = fr.utils.to_numpy(mesh[axis][key[axis]])
        return dims, coords

    @property
    def xr(self) -> xr.DataArray:  # noqa: D102
        return self.xrs[:]

    @property
    def xrs(self) -> fr.utils.SliceableAttribute[xr.DataArray]:  # noqa: D102
        return fr.utils.SliceableAttribute(self._convert_slice_to_xarray)

    @classmethod
    def from_xarray(cls,  # noqa: D102
                    mset: fr.ModelSettingsBase,
                    ds: xr.DataArray,
                    ) -> ScalarField:

        conf = fr.config
        # load the metadata
        mdata = fr.FieldMetadata.from_serializable(ds.attrs)
        # convert the array to backend
        arr = ds.to_numpy().T
        if mdata.is_spectral:
            arr_real = conf.ncp.array(arr["r"])
            arr_imag = conf.ncp.array(arr["i"])
            arr = conf.ncp.array(arr_real + 1j * arr_imag, dtype=conf.dtype_comp)
        else:
            arr = conf.ncp.array(arr, dtype=conf.dtype_real)
            # pad the array
            arr = mset.grid.pad(arr)
        # create the ScalarField
        field = cls(mset=mset, mdata=mdata, arr=arr)
        # synchronize the field
        return field.sync()

    @classmethod
    def from_netcdf(cls,  # noqa: D102
                    mset: fr.ModelSettingsBase,
                    path: str) -> ScalarField:
        import xarray as xr
        ds = xr.open_dataset(path)
        return cls.from_xarray(mset, ds)

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key: slice | tuple[slice | int]) -> ndarray:
        return self.arr[key]

    def __setitem__(self,
                    key: slice | tuple[slice | int],
                    value: ndarray | float) -> None:
        new_arr = fr.utils.modify_array(self.arr, key, value)
        self.arr = new_arr

    # ================================================================
    #  Pickling
    # ================================================================

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ================================================================
    #  Properties
    # ================================================================

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

    @property
    def arr(self) -> ndarray:
        """The underlying array."""
        return self._arr

    @arr.setter
    def arr(self, arr: ndarray) -> None:
        self._arr = arr

    @property
    def mdata(self) -> fr.FieldMetadata:
        """The metadata of the ScalarField."""
        return self._mdata

    @mdata.setter
    def mdata(self, mdata: fr.FieldMetadata) -> None:
        self._mdata = mdata

    @property
    def name(self) -> str:
        """The name of the ScalarField."""
        return self.mdata.name

    @name.setter
    def name(self, name: str) -> None:
        self.mdata.name = name

    @property
    def long_name(self) -> str:
        """The long name of the ScalarField."""
        return self.mdata.long_name

    @long_name.setter
    def long_name(self, long_name: str) -> None:
        self.mdata.long_name = long_name

    @property
    def units(self) -> str:
        """The unit of the ScalarField."""
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
        """True if the ScalarField is in spectral space."""
        return self.mdata.is_spectral

    @property
    def topo(self) -> tuple[bool]:
        """
        Topology of the ScalarField.

        Description
        -----------
        Field Variables do not have to be extended in all directions. For
        example, one might want to create a 2D forcing field for a 3D simulation,
        that only depends on x and y. In this case, the topo of the ScalarField
        would be (True, True, False).
        """
        return self.mdata.topo

    @property
    def position(self) -> fr.grid.Position:
        """The position of the ScalarField on the staggered grid."""
        return self.mdata.position

    @position.setter
    def position(self, position: fr.grid.Position) -> None:
        self.mdata.position = position

    @property
    def bc_types(self) -> tuple[fr.grid.BCType] | None:
        """The boundary condition types for the ScalarField."""
        return self.mdata.bc_types

    @bc_types.setter
    def bc_types(self, bc_types: tuple[fr.grid.BCType] | None) -> None:
        self.mdata.bc_types = bc_types

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the ScalarField."""
        return self.mdata.flags

    @flags.setter
    def flags(self, flags: dict) -> None:
        self.mdata.flags = flags

    # ================================================================
    #  Arithmetic operations
    # ================================================================

    def abs(self) -> ScalarField:
        """Absolute values of the ScalarField."""
        arr = fr.config.ncp.abs(self.arr)
        return ScalarField(mset=self.mset, mdata=deepcopy(self.mdata), arr=arr)

    def __abs__(self) -> ScalarField:
        return self.abs()

    def sum(self, axes: tuple[int] | None = None) -> ScalarField | float:
        """
        Sum of the ScalarField over the whole domain in the specified axes.

        Description
        -----------
        This method computes the sum of the ScalarField over the whole domain
        (across all processes) in the specified axes. If no axes are specified,
        the sum is computed over all axes and a scalar (float) is returned.
        If axes are specified, the sum is computed over the specified axes and
        a new ScalarField that is shrinked in the specified axes is returned.

        Parameters
        ----------
        axes : tuple[int] | None
            The axes to sum over. If None, sum over all axes.

        Returns
        -------
        ScalarField | float
            The sum of the ScalarField over the specified axes.

        """
        # TODO(Silvano): This should call the grid.sum method
        domain = self.grid.domain_decomp
        return domain.sum(self.arr, axes=axes, spectral=self.is_spectral)

    def __sum__(self) -> float:
        return self.sum()

    def max(self, axes: tuple[int] | None = None) -> float:
        """Maximum value of the ScalarField over the whole domain."""
        domain = self.grid.domain_decomp
        return domain.max(self.arr, axes=axes, spectral=self.is_spectral)

    def __max__(self) -> float:
        return self.max()

    def min(self, axes: tuple[int] | None = None) -> float:
        """Minimum value of the ScalarField over the whole domain."""
        domain = self.grid.domain_decomp
        return domain.min(self.arr, axes=axes, spectral=self.is_spectral)

    def __min__(self) -> float:
        return self.min()

    def integrate(self) -> float:
        """Global integral of the ScalarField."""
        if self.is_spectral:
            msg = "Integration not available for spectral fields"
            raise NotImplementedError(msg)
        domain = self.grid.domain_decomp
        return domain.sum(self.arr * self.grid.dV)

    def norm_l2(self) -> float:
        """Compute the numpy.linalg.norm of the ScalarField."""
        norm = fr.config.ncp.linalg.norm(self.unpad())**2
        return fr.config.ncp.sqrt(norm)

    def dot(self,  # noqa: D102
            other: ScalarField | fr.VectorField | fr.TensorField,
            ) -> ScalarField | fr.VectorField | fr.TensorField:
        # check that the spectral flag is the same
        if self.is_spectral != other.is_spectral:
            msg = "Cannot take dot product of spectral and real fields"
            raise ValueError(msg)
        # complex conjugate the other field if it is spectral
        if other.is_spectral:
            other = other.conj()
        # compute the dot product
        return other * self

    def conj(self) -> ScalarField:  # noqa: D102
        return ScalarField(mset=self.mset,
                           mdata=deepcopy(self.mdata),
                           arr=self.arr.conj())

    @staticmethod
    def _apply_operation(
        op: callable, field: ScalarField, other: any) -> ScalarField:
        new_mdata = deepcopy(field.mdata)
        if isinstance(other, ScalarField):
            topo = [p or q for p, q in zip(field.topo, other.topo)]
            new_mdata.topo = topo
            result = op(field.arr, other.arr)
        else:
            result = op(field.arr, other)

        return ScalarField(mset=field.mset, mdata=new_mdata, arr=result)
