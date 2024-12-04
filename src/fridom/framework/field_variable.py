"""field_variable.py - FieldVariable class for the fridom framework."""
from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

import fridom.framework as fr
from fridom.framework.grid.fft_padding import FFTPadding

if TYPE_CHECKING:
    import xarray as xr


@partial(fr.utils.jaxify, dynamic=("_arr", "_position"))
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
    name : str
        Name of the FieldVariable
    position : fr.grid.Position (default cell_center)
        Position of the FieldVariable on the grid
    is_spectral : bool
        True if the FieldVariable should be initialized in spectral space
    topo : list[bool] (default None)
        Topology of the FieldVariable. If None, the FieldVariable is
        assumed to be fully extended in all directions. If a list of booleans
        is given, the FieldVariable has no extend in the directions where the
        corresponding entry is False.
    bc_types : tuple[BCType] (default None)
        Tuple of BCType objects that specify the type of boundary condition
        in each direction. If None, the default boundary conditions is Neumann.
    arr : ndarray (default None)
        The array to be wrapped

    """

    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 name: str,
                 position: fr.grid.Position | None = None,
                 arr: ndarray | None = None,
                 long_name: str = "Unnamed",
                 units: str = "n/a",
                 nc_attrs: dict | None = None,
                 is_spectral: bool = False,
                 topo: list[bool] | None = None,
                 flags: dict | list | None = None,
                 bc_types: tuple[fr.grid.BCType] | None = None,
                 ) -> None:

        # shortcuts
        ncp = fr.config.ncp
        dtype = fr.config.dtype_comp if is_spectral else fr.config.dtype_real

        # position
        position = position or mset.grid.cell_center

        # Topology
        topo = topo or [True] * mset.grid.n_dims

        # Boundary conditions
        bc_types = bc_types or [fr.grid.BCType.NEUMANN] * mset.grid.n_dims

        # The underlying array
        if arr is None:
            data = mset.grid.create_array(
                pad=True,
                spectral=is_spectral,
                topo=tuple(topo),
                )
        else:
            data = ncp.array(arr, dtype=dtype)

        # ----------------------------------------------------------------
        #  Set flags
        # ----------------------------------------------------------------
        self.flags = {"NO_ADV": False,
                      "ENABLE_MIXING": False,
                      "ENABLE_FRICTION": False}
        if isinstance(flags, dict):
            self.flags.update(flags)
        elif isinstance(flags, list):
            for flag in flags:
                if flag not in self.flags:
                    fr.log.warning(f"Flag {flag} not available")
                    fr.log.warning(f"Available flags: {self.flags}")
                    raise ValueError
                self.flags[flag] = True

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------

        self._arr = data
        self._name = name
        self._long_name = long_name
        self._units = units
        self._nc_attrs = nc_attrs or {}
        self._is_spectral = is_spectral
        self._topo = topo
        self._position = position
        self._bc_types = tuple(bc_types)
        self._mset = mset

    def get_kw(self) -> dict:
        """Return keyword arguments for the FieldVariable constructor."""
        return {"mset": self._mset,
                "name": self._name,
                "position": self._position,
                "long_name": self._long_name,
                "units": self._units,
                "nc_attrs": self._nc_attrs,
                "is_spectral": self._is_spectral,
                "topo": self._topo,
                "bc_types": self._bc_types,
                "flags": self._flags}

    def fft(self,
            padding: FFTPadding = FFTPadding.NOPADDING) -> FieldVariable:
        """
        Fourier transform of the FieldVariable.

        If the FieldVariable is already in spectral space, the inverse
        Fourier transform is returned.

        Returns:
            FieldVariable: Fourier transform of the FieldVariable

        """
        if not self.grid.fourier_transform_available:
            message = "Fourier transform not available for this grid"
            raise NotImplementedError(message)

        ncp = fr.config.ncp
        if self.is_spectral:
            res = ncp.array(
                self.grid.ifft(
                    arr=self.arr,
                    padding=padding,
                    bc_types=self.bc_types,
                    positions=self.position.positions).real,
                dtype=fr.config.dtype_real)
        else:
            res = ncp.array(
                self.grid.fft(
                    arr=self.arr,
                    padding=padding,
                    bc_types=self.bc_types,
                    positions=self.position.positions),
                dtype=fr.config.dtype_comp)
        f = copy(self)
        f.arr = res
        f._is_spectral = not self.is_spectral

        return f

    def ifft(self,
             padding: FFTPadding = FFTPadding.NOPADDING) -> FieldVariable:
        """Inverse Fourier transform of the FieldVariable."""
        if not self.is_spectral:
            message = "FieldVariable is not in spectral space, cannot perform ifft"
            raise ValueError(message)
        return self.fft(padding=padding)

    def sync(self) -> FieldVariable:
        """Synchronize the FieldVariable (exchange boundary values)."""
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
            `[df/dx, None, df/dz].

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
        return FieldVariable(arr=deepcopy(self.arr),
                             **self.get_kw())

    def __deepcopy__(self, memo: dict) -> FieldVariable:
        return FieldVariable(arr=deepcopy(self.arr, memo),
                             **deepcopy(self.get_kw(), memo))

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
    def xrs(self) -> fr.utils.SliceableAttribute:
        """
        Convert a slice of the FieldVariable to xarray DataArray.

        Example:
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

            all_attrs = deepcopy(fv.nc_attrs)
            all_attrs.update({"long_name": fv.long_name, "units": fv.units})

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

    # ==================================================================
    #  OTHER METHODS
    # ==================================================================

    def has_nan(self) -> bool:
        """Check if the FieldVariable contains NaN values."""
        return fr.config.ncp.any(fr.config.ncp.isnan(self.arr))

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
    def name(self) -> str:
        """The name of the FieldVariable."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def long_name(self) -> str:
        """The long name of the FieldVariable."""
        return self._long_name

    @long_name.setter
    def long_name(self, long_name: str) -> None:
        self._long_name = long_name

    @property
    def units(self) -> str:
        """The unit of the FieldVariable."""
        return self._units

    @units.setter
    def units(self, units: str) -> None:
        self._units = units

    @property
    def nc_attrs(self) -> dict:
        """Dictionary with additional attributes for the NetCDF file or xarray."""
        return self._nc_attrs

    @nc_attrs.setter
    def nc_attrs(self, nc_attrs: dict) -> None:
        self._nc_attrs = nc_attrs

    @property
    def is_spectral(self) -> bool:
        """True if the FieldVariable is in spectral space."""
        return self._is_spectral

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
        return self._topo

    @property
    def position(self) -> fr.grid.Position:
        """The position of the FieldVariable on the staggered grid."""
        return self._position

    @position.setter
    def position(self, position: fr.grid.Position) -> None:
        self._position = position

    @property
    def bc_types(self) -> tuple[fr.grid.BCType] | None:
        """The boundary condition types for the FieldVariable."""
        return self._bc_types

    @bc_types.setter
    def bc_types(self, bc_types: tuple[fr.grid.BCType] | None) -> None:
        self._bc_types = bc_types

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the FieldVariable."""
        return self._flags

    @flags.setter
    def flags(self, flags: dict) -> None:
        self._flags = flags

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
        return FieldVariable(arr=fr.config.ncp.abs(self.arr), **self.get_kw())

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
        kwargs = field.get_kw()
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(field.topo, other.topo)]
            kwargs["topo"] = topo
            result = op(field.arr, other.arr)
        else:
            result = op(field.arr, other)

        return FieldVariable(arr=result, **kwargs)

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
        return FieldVariable(arr=-self.arr, **self.get_kw())
