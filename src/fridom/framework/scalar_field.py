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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()

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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()

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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # synchronize the array
        self.arr = self.grid.sync(self.arr)
        self.apply_water_mask()
        return self

    def apply_water_mask(self) -> ScalarField:  # noqa: D102
        # the water mask is defined on the 3D grid so we can't apply it
        # for non full domain fields
        # TODO(Silvano): Maybe we can assign custom water masks for scalar fields
        # so that we can apply them to non full domain fields
        self._check_full_domain()
        self._check_not_spectral()
        self.arr *= self.grid.water_mask.get_mask(self.position)
        return self

    def has_nan(self) -> bool:  # noqa: D102
        ncp = fr.config.ncp
        return ncp.any(ncp.isnan(self.arr))

    def set_random(self, seed: int = 1234) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # create the random array and set it
        self.arr = self.grid.create_random_array(seed=seed, spectral=self.is_spectral)
        return self

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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Make this work for spectral fields
        if self.is_spectral:
            msg = "Cannot unpad spectral field"
            raise ValueError(msg)
        return self.grid.unpad(self.arr)

    def get_mesh(self) -> tuple[ndarray]:
        """
        Get the meshgrid of the ScalarField.

        Description
        -----------
        This method returns the meshgrid of the ScalarField. It returns a tuple
        of ndarrays, where each ndarray represents the meshgrid in one direction.
        For example, a 3D field that is extended in x, z but not in y would return
        a tuple of 2 ndarrays (x, z).

        Returns
        -------
        tuple[ndarray]
            The meshgrid of the ScalarField for each direction that is extended.

        """
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Make this work for spectral fields
        self._check_not_spectral()
        return self.grid.interp_module.interpolate(self, destination)

    # ================================================================
    #  Check Methods (for internal use)
    # ================================================================

    def _check_full_domain(self) -> None:
        if not all(self.topo):
            msg = "Operation not available for non full domain fields"
            raise NotImplementedError(msg)

    def _check_not_spectral(self) -> None:
        if self.is_spectral:
            msg = "Operation not available for spectral fields"
            raise NotImplementedError(msg)

    def _check_axes_argument(self, axes: tuple[int] | None) -> None:
        if axes is not None:
            msg = "Operation not available for specific axes"
            raise NotImplementedError(msg)

    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self, axis: int, order: int = 1) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Make this work for spectral fields
        self._check_not_spectral()
        return self.grid.diff_module.diff(self, axis, order)

    def grad(self, axes: list[int] | None = None ) -> fr.VectorField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Make this work for spectral fields
        self._check_not_spectral()
        return self.grid.diff_module.grad(self, axes)

    def laplacian(self,  # noqa: D102
                  axes: tuple[int] | None = None,
                  ) -> ScalarField:
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Make this work for spectral fields
        self._check_not_spectral()
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
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
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

        # add the slice key as an attribute
        dv.attrs["slice_key"] = str(key)
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
        # read in the slice key
        # in general, eval poses a security risk, we eliminate this risk by
        # setting the __builtins__ to None and only allowing the slice function
        # to be used
        # this means that statements like "__import__('os')" will not work
        slice_key = eval(ds.attrs["slice_key"],  # noqa: S307
                         {"__builtins__": None},
                         {"slice": slice})
        # TODO(Silvano): Add option to read from sliced dataarrays
        # This could for example be implemented by creating the full array
        # and then setting the slice region to the values of the dataarray
        # another option would be to allow for local fields that lives in a subregion
        # of the domain. For now, we don't allow for sliced dataarrays to be
        # converted back to ScalarFields
        if not isinstance(slice_key, tuple):
            slice_key = (slice_key,)
        for key in slice_key:
            if key != slice(None):
                msg = "Cannot convert sliced dataarray to ScalarField"
                raise ValueError(msg)

        # load the metadata
        mdata = fr.FieldMetadata.from_serializable(ds.attrs)
        # convert the array to backend
        arr = ds.to_numpy().T
        # if the array is loaded with xarray from a netcdf file, complex arrays
        # are stored as two separate arrays for the real and imaginary part
        # we check if the array has a "r" and "i" key and if so, we combine them
        try:
            arr["r"]
            separate = True
        except IndexError:
            separate = False
        if separate:
            arr_real = conf.ncp.array(arr["r"])
            arr_imag = conf.ncp.array(arr["i"])
            arr = conf.ncp.array(arr_real + 1j * arr_imag, dtype=conf.dtype_comp)
        else:
            dtype = conf.dtype_comp if mdata.is_spectral else conf.dtype_real
            arr = conf.ncp.array(arr, dtype=dtype)

        if not mdata.is_spectral:
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
        ds = xr.open_dataarray(path)
        return cls.from_xarray(mset, ds)

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key: slice | tuple[slice | int]) -> ScalarField:
        msg = "Slicing is currently not supported for ScalarFields"
        raise NotImplementedError(msg)

    def __setitem__(self,
                    key: slice | tuple[slice | int],
                    value: ScalarField | float) -> None:
        msg = "Slicing is currently not supported for ScalarFields"
        raise NotImplementedError(msg)

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
    #  Shrink / Extend operations
    # ================================================================

    def extend(self, topo: tuple[bool]) -> ScalarField:  # noqa: D102
        # check if the topology is valid (no shrinking)
        old_topo = self.topo
        for (old, new) in zip(old_topo, topo):
            if old and not new:
                msg = "Cannot shrink the field in any direction"
                raise ValueError(msg)
        # TODO(Silvano): The grid.extend method is not implemented yet
        msg = "The grid.extend method is not implemented yet"
        raise NotImplementedError(msg)
        return self.grid.extend(self, topo)

    def _set_shrinked_field(self, arr: ndarray, axes: tuple[int] | None) -> ScalarField:
        """Shrink the ScalarField in the specified axes and set the new array."""
        if axes is None:
            axes = tuple(i for i in range(self.grid.n_dims))
        new_mdata = deepcopy(self.mdata)
        # set the new topo
        topo = list(new_mdata.topo)
        for axis in axes:
            topo[axis] = False
        new_mdata.topo = tuple(topo)
        return ScalarField(mset=self.mset, mdata=new_mdata, arr=arr)

    def sum(self, axes: tuple[int] | None = None) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Implement sum over specific axes
        self._check_axes_argument(axes)
        # TODO(Silvano): This should call the grid.sum method
        domain = self.grid.domain_decomp
        result = domain.sum(self.arr, axes=axes, spectral=self.is_spectral)
        # result must be a n-dimensional array
        shape = tuple([1] * self.grid.n_dims)
        result = fr.config.ncp.full(shape, result)
        return self._set_shrinked_field(arr=result, axes=axes)

    def max(self, axes: tuple[int] | None = None) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Implement max over specific axes
        self._check_axes_argument(axes)
        # TODO(Silvano): This should call the grid.max method
        domain = self.grid.domain_decomp
        result = domain.max(self.arr, axes=axes, spectral=self.is_spectral)
        # result must be a n-dimensional array
        shape = tuple([1] * self.grid.n_dims)
        result = fr.config.ncp.full(shape, result)
        return self._set_shrinked_field(arr=result, axes=axes)

    def min(self, axes: tuple[int] | None = None) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Implement min over specific axes
        self._check_axes_argument(axes)
        # TODO(Silvano): This should call the grid.min method
        domain = self.grid.domain_decomp
        result = domain.min(self.arr, axes=axes, spectral=self.is_spectral)
        # result must be a n-dimensional array
        shape = tuple([1] * self.grid.n_dims)
        result = fr.config.ncp.full(shape, result)
        return self._set_shrinked_field(arr=result, axes=axes)

    def integrate(self, axes: tuple[int] | None = None) -> ScalarField:  # noqa: D102
        # TODO(Silvano): Make this work for non full domain fields
        self._check_full_domain()
        # TODO(Silvano): Implement integration over specific axes
        self._check_axes_argument(axes)
        # TODO(Silvano): The sum method cannot be used here because of the
        # ghost cells. We need to implement a proper integration method
        # that takes the grid spacing into account
        msg = "Integration is not implemented yet"
        raise NotImplementedError(msg)

    # ================================================================
    #  Arithmetic operations
    # ================================================================

    def abs(self) -> ScalarField:
        """Absolute values of the ScalarField."""
        arr = fr.config.ncp.abs(self.arr)
        return ScalarField(mset=self.mset, mdata=deepcopy(self.mdata), arr=arr)

    def __abs__(self) -> ScalarField:
        return self.abs()

    def norm_l2(self) -> float:
        r"""
        Compute the L2 norm of the ScalarField.

        .. math::
            ||f||_{L2} = \sqrt{\int_{\Omega} |f(\boldsymbol{x})|^2 d\boldsymbol{x}}

        """
        # integrate the square of the field
        return (abs(self)**2).integrate() ** 0.5

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
    def _apply_operation(op: callable,
                         field: ScalarField,
                         other: ScalarField | complex | np.number,
                         ) -> ScalarField:
        new_mdata = deepcopy(field.mdata)
        if isinstance(other, ScalarField):
            topo = [p or q for p, q in zip(field.topo, other.topo)]
            new_mdata.topo = topo
            result = op(field.arr, other.arr)
        elif isinstance(other, (int, float, complex, np.number)):
            result = op(field.arr, other)
        else:
            return NotImplemented

        return ScalarField(mset=field.mset, mdata=new_mdata, arr=result)
