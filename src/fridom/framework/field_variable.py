import fridom.framework as fr
from typing import TYPE_CHECKING
from copy import deepcopy
from mpi4py import MPI
from numpy import ndarray
import numpy as np

if TYPE_CHECKING:
    import xarray as xr


class FieldVariable:
    """
    Class for field variables in the framework.
    
    Description
    -----------
    TODO

    Parameters
    ----------
    `mset` : `ModelSettings`
        ModelSettings object
    `is_spectral` : `bool`
        True if the FieldVariable should be initialized in spectral space
    `name` : `str` 
        Name of the FieldVariable
    `topo` : `list[bool]` (default None)
        Topology of the FieldVariable. If None, the FieldVariable is
        assumed to be fully extended in all directions. If a list of booleans 
        is given, the FieldVariable has no extend in the directions where the
        corresponding entry is False.
    `transform_types` : `tuple[TransformType]` (default None)
        Tuple of TransformType objects that specify the type of transform
        that should be applied to nonperiodic axes (e.g. DST1, DST2, etc.).
        If None, the default transform type DCT2 is used.
    `arr` : `ndarray` (default None)
        The array to be wrapped
    
    """
    _dynamic_attributes = set(["_arr", "_position"])
    def __init__(self, 
                 mset: fr.ModelSettingsBase,
                 name: str,
                 position: fr.grid.Position,
                 arr: ndarray | None = None,
                 long_name: str = "Unnamed", 
                 units: str = "n/a",
                 nc_attrs: dict | None = None,
                 is_spectral: bool = False, 
                 topo: list[bool] | None = None,
                 flags: dict | list | None = None,
                 transform_types: tuple[fr.grid.TransformType] | None = None,
                 ) -> None:

        # shortcuts
        ncp = fr.config.ncp
        dtype = fr.config.dtype_comp if is_spectral else fr.config.dtype_real

        # Topology
        topo = topo or [True] * mset.grid.n_dims

        # The underlying array
        if arr is None:
            # get the shape of the array
            if is_spectral:
                shape = tuple(n if p else 1 
                              for n, p in zip(mset.grid.K[0].shape, topo))
            else:
                shape = tuple(n if p else 1 
                              for n, p in zip(mset.grid.X[0].shape, topo))
            data = ncp.zeros(shape=shape, dtype=dtype)
        else:
            data = ncp.array(arr, dtype=dtype)

        # ----------------------------------------------------------------
        #  Set flags
        # ----------------------------------------------------------------
        self.flags = {"NO_ADV": False, 
                      "ENABLE_MIXING": False}
        if isinstance(flags, dict):
            self.flags.update(flags)
        elif isinstance(flags, list):
            for flag in flags:
                if flag not in self.flags:
                    fr.config.logger.warning(f"Flag {flag} not available")
                    fr.config.logger.warning(f"Available flags: {self.flags}")
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
        self._transform_types = transform_types
        self._mset = mset
        return

    def get_kw(self):
        """
        Return a dictionary with the keyword arguments for the
        FieldVariable constructor
        """
        return {"mset": self._mset, 
                "name": self._name,
                "position": self._position,
                "long_name": self._long_name,
                "units": self._units,
                "nc_attrs": self._nc_attrs,
                "is_spectral": self._is_spectral, 
                "topo": self._topo,
                "flags": self._flags,
                "transform_types": self._transform_types}

    def fft(self) -> "FieldVariable":
        """
        Fourier transform of the FieldVariable

        If the FieldVariable is already in spectral space, the inverse
        Fourier transform is returned.

        Returns:
            FieldVariable: Fourier transform of the FieldVariable
        """
        if not self.grid.fourier_transform_available:
            raise NotImplementedError(
                "Fourier transform not available for this grid")

        ncp = fr.config.ncp
        if self.is_spectral:
            res = ncp.array(
                self.grid.ifft(self.arr, self.transform_types).real, 
                dtype=fr.config.dtype_real)
        else:
            res = ncp.array(
                self.grid.fft(self.arr, self.transform_types),
                dtype=fr.config.dtype_comp)
        from copy import copy
        f = copy(self)
        f.arr = res
        f._is_spectral = not self.is_spectral

        return f

    def sync(self) -> 'FieldVariable':
        """
        Synchronize the FieldVariable (exchange boundary values)
        """
        self.arr = self.grid.sync(self.arr)
        return self


    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self, axis: int, order: int = 1) -> 'FieldVariable':
        r"""
        Compute the partial derivative along an axis.

        .. math::
            \partial_i^n f

        with axis :math:`i` and order :math:`n`.

        Parameters
        ----------
        `axis` : `int`
            The axis along which to differentiate.
        `order` : `int`
            The order of the derivative. Default is 1.

        Returns
        -------
        `FieldVariable`
            The derivative of the field along the specified axis.
        """
        return self.grid.diff_mod.diff(self, axis, order)

    def grad(self, axes: list[int] | None = None ) -> 'tuple[FieldVariable | None]':
        r"""
        Compute the gradient.

        .. math::
            \nabla f = 
            \begin{pmatrix} \partial_1 f \\ \dots \\ \partial_n f \end{pmatrix}

        Parameters
        ----------
        `axes` : `list[int] | None` (default is None)
            The axes along which to compute the gradient. If `None`, the
            gradient is computed along all axes.

        Returns
        -------
        `tuple[FieldVariable | None]`
            The gradient of the field along the specified axes. The list contains 
            the gradient components along each axis. Axis which are not included 
            in `axes` will have a value of `None`. 
            E.g. for a 3D grid, `diff.grad(f, axes=[0, 2])` will return
            `[df/dx, None, df/dz]`.
        """
        return self.grid.diff_mod.grad(self, axes)

    def laplacian(self, 
                  axes: tuple[int] | None = None
                  ) -> 'FieldVariable':
        r"""
        Compute the Laplacian.

        .. math::
            \nabla^2 f = \sum_{i=1}^n \partial_i^2 f

        Parameters
        ----------
        `axes` : `tuple[int] | None` (default is None)
            The axes along which to compute the Laplacian. If `None`, the
            Laplacian is computed along all axes.

        Returns
        -------
        `FieldVariable`
            The Laplacian of the field.
        """
        return self.grid.diff_mod.laplacian(self, axes)

    def curl(self,
             arrs: 'list[ndarray]',
             axes: list[int] | None = None,
             **kwargs) -> 'list[ndarray]':
            """
            Calculate the curl of a vector field (\\nabla \\times \\vec{v}).
    
            Parameters
            ----------
            `arrs` : `list[ndarray]`
                The list of arrays representing the vector field.
            `axes` : `list[int]` or `None` (default: `None`)
                The axes along which to compute the curl. If `None`, the
                curl is computed along all axes.
    
            Returns
            -------
            `list[ndarray]`
                The curl of the vector field.
            """
            return self._diff_mod.curl(arrs, axes, **kwargs)

    def interpolate(self, destination: fr.grid.Position) -> 'FieldVariable':
        """
        Interpolate the field to the destination position.
        
        Parameters
        ----------
        `destination` : `fr.grid.Position`
            The position to interpolate to.
        
        Returns
        -------
        `FieldVariable`
            The interpolated field.
        """
        return self.grid.interp_mod.interpolate(self, destination)

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key) -> ndarray:
        return self.arr[key]
    
    def __setitem__(self, key, value):
        new_arr = fr.utils.modify_array(self.arr, key, value)
        self.arr = new_arr

    def __getattr__(self, name):
        """
        Forward attribute access to the underlying array (e.g. shape)
        """
        try:
            return getattr(self.arr, name)
        except AttributeError:
            raise AttributeError(f"FieldVariable has no attribute {name}")

    # ================================================================
    #  Pickling
    # ================================================================

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo: dict) -> 'FieldVariable':
        return FieldVariable(arr=deepcopy(self.arr, memo), 
                             **deepcopy(self.get_kw(), memo))

    # ==================================================================
    #  Display methods
    # ==================================================================

    @property
    def info(self) -> dict:
        """
        Dictionary with information about the field.
        """
        res = {}
        res["name"] = self.name
        res["long_name"] = self.long_name
        res["units"] = self.units
        res["is_spectral"] = self.is_spectral
        res["position"] = self.position
        res["topo"] = self.topo
        res["transform_types"] = self.transform_types
        enabled_flags = [key for key, value in self.flags.items() if value]
        res["enabled_flags"] = enabled_flags
        return res

    def __repr__(self) -> str:
        res = "FieldVariable"
        for key, value in self.info.items():
            res += "\n  - {}: {}".format(key, value)
        return res

    # ================================================================
    #  xarray conversion
    # ================================================================

    @property
    def xr(self) -> 'xr.DataArray':
        """Convert to xarray DataArray"""
        return self.xrs[:]

    @property
    def xrs(self) -> fr.utils.SliceableAttribute:
        """
        Convert a slice of the FieldVariable to xarray DataArray

        Example
        -------
        Let `f` be a large 3D FieldVariable and we want to convert the top 
        of the field to an xarray DataArray. To avoid loading the whole field 
        into memory, we can use slicing:

        .. code-block:: python

            data_array = f.xrs[:,:,-1]  # Only the top of the field
        """
        def slicer(key):
            import xarray as xr
            fv = self
            # convert key to tuple
            ndim = fv.grid.n_dims
            if not isinstance(key, (tuple, list)):
                key = [key]
            else:
                key = list(key)
            key += [slice(None)] * (ndim - len(key))

            # get the inner of the field
            if fv.is_spectral:
                # no inner slice for spectral fields
                ics = [slice(None)] * ndim
            else:
                ics = list(fv.grid.inner_slice)

            for i in range(ndim):
                # set non-extended axes to 0
                if not fv.topo[i]:
                    ics[i] = slice(0,1)
                    key[i] = slice(0,1)
                if isinstance(key[i], int):
                    key[i] = slice(key[i], key[i]+1)

            arr = fr.utils.to_numpy(fv.arr[tuple(ics)][tuple(key)])

            # get the coordinates
            if ndim <= 3:
                if fv.is_spectral:
                    all_dims = tuple(["kx", "ky", "kz"][:ndim])
                else:
                    all_dims = tuple(["x", "y", "z"][:ndim])
            else:
                if fv.is_spectral:
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
                    x_sel = fv.grid.k_local[axis][key[axis]]
                else:
                    x_sel = fv.grid.x_local[axis][key[axis]]
                coords[dim] = fr.utils.to_numpy(x_sel)

            # reverse the dimensions
            dims = dims[::-1]

            all_attrs = deepcopy(fv.nc_attrs)
            all_attrs.update({"long_name": fv.long_name, "units": fv.units})
    
            dv = xr.DataArray(
                np.squeeze(arr).T, 
                coords=coords, 
                dims=tuple(dims),
                name=fv.name,
                attrs=all_attrs)
    
            if fv.is_spectral:
                x_unit = "1/m"
            else:
                x_unit = "m"
            for dim in dims:
                dv[dim].attrs["units"] = x_unit
            return dv
        return fr.utils.SliceableAttribute(slicer)

    # ==================================================================
    #  OTHER METHODS
    # ==================================================================

    def has_nan(self) -> bool:
        """Check if the FieldVariable contains NaN values"""
        return fr.config.ncp.any(fr.config.ncp.isnan(self.arr))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def arr(self) -> ndarray:
        """The underlying array"""
        return self._arr

    @arr.setter
    def arr(self, arr: ndarray):
        self._arr = arr

    @property
    def name(self) -> str:
        """The name of the FieldVariable"""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def long_name(self) -> str:
        """The long name of the FieldVariable"""
        return self._long_name

    @long_name.setter
    def long_name(self, long_name: str):
        self._long_name = long_name
    
    @property
    def units(self) -> str:
        """The unit of the FieldVariable"""
        return self._units

    @units.setter
    def units(self, units: str):
        self._units = units
    
    @property
    def nc_attrs(self) -> dict:
        """Dictionary with additional attributes for the NetCDF file or xarray"""
        return self._nc_attrs

    @nc_attrs.setter
    def nc_attrs(self, nc_attrs: dict):
        self._nc_attrs = nc_attrs

    @property
    def is_spectral(self) -> bool:
        """True if the FieldVariable is in spectral space"""
        return self._is_spectral

    @property
    def topo(self) -> list[bool]:
        """Topology of the FieldVariable
        
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
        """The position of the FieldVariable on the staggered grid"""
        return self._position

    @position.setter
    def position(self, position: fr.grid.Position):
        self._position = position

    @property
    def transform_types(self) -> tuple[fr.grid.TransformType] | None:
        """The transform types for nonperiodic axes"""
        return self._transform_types

    @transform_types.setter
    def transform_types(self, transform_types: tuple[fr.grid.TransformType] | None):
        self._transform_types = transform_types

    @property
    def flags(self) -> dict:
        """Dictionary with flag options for the FieldVariable"""
        return self._flags

    @flags.setter
    def flags(self, flags: dict):
        self._flags = flags

    @property
    def mset(self) -> fr.ModelSettingsBase:
        """The model settings object"""
        return self._mset

    @property
    def grid(self):
        """The grid object"""
        return self._mset.grid

    # ==================================================================
    #  ARITHMETIC OPERATIONS
    # ==================================================================

    def abs(self) -> 'FieldVariable':
        """Absolute values of the FieldVariable"""
        return FieldVariable(arr=fr.config.ncp.abs(self.arr), **self.get_kw())

    def __abs__(self) -> 'FieldVariable':
        return self.abs()

    def sum(self) -> float:
        """Global sum of the FieldVariable"""
        ics = self.grid.inner_slice
        sum = self.arr[ics].sum()
        sum = MPI.COMM_WORLD.allreduce(sum, op=MPI.SUM)
        return sum

    def __sum__(self) -> float:
        return self.sum()
    
    def max(self) -> float:
        """Maximum value of the FieldVariable over the whole domain"""
        ics = self.grid.inner_slice
        my_max = self.arr[ics].max()
        return MPI.COMM_WORLD.allreduce(my_max, op=MPI.MAX)

    def __max__(self) -> float:
        return self.max()
    
    def min(self) -> float:
        """Minimum value of the FieldVariable over the whole domain"""
        ics = self.grid.inner_slice
        my_min = self.arr[ics].min()
        return MPI.COMM_WORLD.allreduce(my_min, op=MPI.MIN)
    
    def __min__(self) -> float:
        return self.min()

    def integrate(self) -> float:
        """Global integral of the FieldVariable"""
        ics = self.grid.inner_slice
        integral = (self.arr * self.grid.dV)[ics].sum()
        return MPI.COMM_WORLD.allreduce(integral, op=MPI.SUM)

    def norm_l2(self) -> float:
        """Compute the numpy.linalg.norm of the FieldVariable"""
        ics = self.grid.inner_slice
        local_norm = fr.config.ncp.linalg.norm(self.arr[ics])**2
        global_norm = MPI.COMM_WORLD.allreduce(local_norm, op=MPI.SUM)
        return fr.config.ncp.sqrt(global_norm)

    def __add__(self, other: any) -> 'FieldVariable':
        """Add something to the FieldVariable"""
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            sum = self.arr + other.arr
        else:
            sum = self.arr + other

        return FieldVariable(arr=sum, **kwargs)
    
    def __radd__(self, other: any) -> 'FieldVariable':
        """Add a FieldVariable to something"""
        return self.__add__(other)

    def __sub__(self, other: any) -> 'FieldVariable':
        """Subtract something from the FieldVariable"""
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            diff = self.arr - other.arr
        else:
            diff = self.arr - other

        return FieldVariable(arr=diff, **kwargs)
    
    def __rsub__(self, other: any) -> 'FieldVariable':
        """Subtract the FieldVariable from something"""
        res = other - self.arr
        return FieldVariable(arr=res, **self.get_kw())

    def __mul__(self, other: any) -> 'FieldVariable':
        """Multiply the FieldVariable with something"""
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            prod = self.arr * other.arr
        else:
            prod = self.arr * other

        return FieldVariable(arr=prod, **kwargs)
    
    def __rmul__(self, other: any) -> 'FieldVariable':
        """Multiply something with the FieldVariable"""
        return self.__mul__(other)
    
    def __truediv__(self, other: any) -> 'FieldVariable':
        """Divide the FieldVariable by something"""
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            quot = self.arr / other.arr
        else:
            quot = self.arr / other

        return FieldVariable(arr=quot, **kwargs)
    
    def __rtruediv__(self, other: any) -> 'FieldVariable':
        """Divide something by the FieldVariable"""
        return FieldVariable(arr=other / self.arr, **self.get_kw())

    def __pow__(self, other: any) -> 'FieldVariable':
        """Raise the FieldVariable to a power"""
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            pow = self.arr ** other.arr
        else:
            pow = self.arr ** other

        return FieldVariable(arr=pow, **kwargs)

    def __neg__(self) -> 'FieldVariable':
        """Negate the FieldVariable"""
        return FieldVariable(arr=-self.arr, **self.get_kw())

fr.utils.jaxify_class(FieldVariable)