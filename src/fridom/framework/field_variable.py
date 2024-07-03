# Import external modules
from typing import TYPE_CHECKING
from copy import deepcopy
# Import internal modules
from fridom.framework import config
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase


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
    `name` : `str` (default "Unnamed")
        Name of the FieldVariable
    `topo` : `list[bool]` (default None)
        Topology of the FieldVariable. If None, the FieldVariable is
        assumed to be fully extended in all directions. If a list of booleans 
        is given, the FieldVariable has no extend in the directions where the
        corresponding entry is False.
    `arr` : `ndarray` (default None)
        The array to be wrapped
    
    Attributes
    ----------
    `name` : `str`
        The name of the FieldVariable
    `long_name` : `str`
        The long name of the FieldVariable
    `units` : `str`
        The unit of the FieldVariable
    `nc_attrs` : `dict`
        Dictionary with additional attributes for the NetCDF file
    `mset` : `ModelSettings`
        ModelSettings object
    `grid` : `Grid`
        Grid object
    `is_spectral` : `bool`
        True if the FieldVariable is in spectral space
    `topo` : `list[bool]`
        Topology of the FieldVariable
    `arr` : `ndarray`
        The underlying array
    
    Methods
    -------
    `fft()`
        Compute forward and backward Fourier transform of the FieldVariable
    `sync()`
        Synchronize the FieldVariable (exchange boundary values)
    `sqrt()`
        Compute the square root of the FieldVariable
    `norm_l2()`
        Compute the L2 norm of the FieldVariable
    `pad_raw(pad_width)`
        Return a padded version of the FieldVariable with the given padding width
    `ave(shift, axis)`
        Compute the average in a given direction
    `diff_forward(axis)`
        Compute the forward difference in a given direction
    `diff_backward(axis)`
        Compute the backward difference in a given direction
    `diff_2(axis)`
        Compute the second order difference in a given direction
    
    
    Examples
    --------
    TODO
    """
    def __init__(self, 
                 mset: 'ModelSettingsBase',
                 is_spectral=False, 
                 name="Unnamed", 
                 long_name="Unnamed", 
                 units="n/a",
                 nc_attrs=None,
                 topo=None,
                 arr=None) -> None:

        ncp = config.ncp
        dtype = config.dtype_comp if is_spectral else config.dtype_real
        topo = topo or [True] * mset.grid.n_dims
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
        #  Set attributes
        # ----------------------------------------------------------------

        self.name = name
        self.long_name = long_name
        self.units = units
        self.nc_attrs = nc_attrs or {}
        self.mset = mset
        self.grid = mset.grid
        self.is_spectral = is_spectral
        self.topo = topo
        self.arr = data

        return
        
    # ==================================================================
    #  OTHER METHODS
    # ==================================================================

    def get_kw(self):
        """
        Return a dictionary with the keyword arguments for the
        FieldVariable constructor
        """
        return {"mset": self.mset, 
                "name": self.name,
                "long_name": self.long_name,
                "units": self.units,
                "nc_attrs": self.nc_attrs,
                "is_spectral": self.is_spectral, 
                "topo": self.topo,}

    def fft(self) -> "FieldVariable":
        """
        Compute forward and backward Fourier transform of the FieldVariable

        Returns:
            FieldVariable: Fourier transform of the FieldVariable
        """
        if not self.grid.fourier_transform_available:
            raise NotImplementedError(
                "Fourier transform not available for this grid")

        ncp = config.ncp
        if self.is_spectral:
            res = ncp.array(self.grid.ifft(self.arr).real, 
                           dtype=config.dtype_real)
        else:
            res = ncp.array(self.grid.fft(self.arr), 
                           dtype=config.dtype_comp)

        kw = self.get_kw()
        kw["is_spectral"] = not self.is_spectral

        return FieldVariable(arr=res, **kw)

    def sync(self):
        """
        Synchronize the FieldVariable (exchange boundary values)
        """
        if not self.grid.mpi_available:
            raise NotImplementedError(
                "MPI not available for this grid")
        if self.is_spectral:
            self.grid.sync_spectral(self.arr)
        else:
            self.grid.sync_physical(self.arr)

    def apply_boundary_conditions(self, axis, side, value):
        """
        Apply boundary conditions to the FieldVariable
        
        Parameters
        ----------
        `axis` : `int`
            Axis along which to apply the boundary condition
        `side` : `str`
            Side of the axis along which to apply the boundary condition
            (either "left" or "right")
        `value` : `float | np.ndarray | FieldVariable`
            The value of the boundary condition. If a float is provided, the
            boundary condition will be set to a constant value. If an array is
            provided, the boundary condition will be set to the array.
        
        Raises
        ------
        `NotImplementedError`
            If the FieldVariable is in spectral space
        """
        if self.is_spectral:
            raise NotImplementedError(
                "Boundary conditions not available in spectral space")

        self.grid.apply_boundary_condition(self.arr, axis, side, value)
        return


    def spectra_1d(self, nbins=50) -> tuple:
        """
        Calculate the 1D spectra of the FieldVariable. The n-dimensional spectra 
        is is integrated over spherical shells in spectral space.

        Arguments:
            nbins (int): Number of bins in the 1D spectra (shell width)
            normalize_with_volume (bool): If True, the 1D spectra is
                normalized with the volume of the spherical shells. If
                False, the 1D spectra is normalized with the volume of
                the grid.

        Returns:
            k_centers (ndarray) : Bin centers of the 1D spectra
            spectra_1D (ndarray): 1D spectra
        """
        ncp = config.ncp
        if self.is_spectral:
            spectral = self.arr.copy()
        else:
            spectral = self.fft().arr.copy()
        # normalize with respect to the number of grid points
        spectral /= spectral.size
        # check if domain was extended
        import numpy
        fac = numpy.prod([1 if p else 2 for p in self.grid.periodic_bounds])
        spectral /= fac

        # calculate n-dimensional spectra and flatten the array
        spectra_3D = (ncp.abs(spectral)**2).flatten()
        # calculate the corresponding wave number
        k = ncp.zeros_like(spectral).real
        for K in self.grid.K:
            k += K**2
        k = ncp.sqrt(k).flatten()

        # find largest grid spacing in spectral space
        dk_max = 0
        k_vol = 1
        for kk in self.grid.k:
            dk_max = max(dk_max, ncp.abs(kk[1] - kk[0]))
            k_vol *= kk[1] - kk[0]

        kmax = float(ncp.max(k))
        dk_max = float(dk_max)
        # find the bin edges
        k_bins = ncp.arange(0, kmax + dk_max, dk_max)
        # find the bin centers
        k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
        # integrate the 3D spectra over the bins
        k_idx = ncp.digitize(k, k_bins)
        spectra_1D = ncp.bincount(k_idx, weights=spectra_3D)[1:]
        # normalize:
        spectra_1D *= k_vol/(k_centers[1] - k_centers[0])

        return k_centers, spectra_1D

    def sqrt(self):
        """
        Compute the square root of the FieldVariable

        Returns:
            FieldVariable: Square root of the FieldVariable
        """
        return FieldVariable(arr=config.ncp.sqrt(self.arr), **self.get_kw())

    def norm_l2(self):
        """
        Compute the L2 norm of the FieldVariable

        Returns:
            float: L2 norm of the FieldVariable
        """
        return config.ncp.linalg.norm(self.arr)

    def pad_raw(self, pad_width):
        """
        Return a padded version of the FieldVariable with the given padding
        width. Padded values are periodic.

        Arguments:
            pad_width (int or tuple): Padding width

        Returns:
            p_arr (ndarray): Padded version of the FieldVariable
        """
        ncp = config.ncp
        p_arr = ncp.pad(self.arr, pad_width, mode="wrap")
        return p_arr

    def ave(self, shift, axis):
        """
        Compute the average in a given direction

        Arguments:
            shift (int): Shift in the given direction [1 or -1]
            axis (int) : Axis along which to compute the average

        Returns:
            float: Average of the FieldVariable
        """
        shift_positive = shift >= 0
        # get padding width
        if shift_positive:
            pad_width = tuple([(0, 0) if i != axis else (0, 1) 
                               for i in range(self.grid.n_dims)])
        else:
            pad_width = tuple([(0, 0) if i != axis else (1, 0) 
                               for i in range(self.grid.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.grid.periodic_bounds[axis]:
            if shift_positive:
                sl = tuple([slice(None) if i != axis else -1
                            for i in range(self.grid.n_dims)])
            else:
                sl = tuple([slice(None) if i != axis else 0
                            for i in range(self.grid.n_dims)])
            p_arr[sl] = 0
                

        # get slices for the center and shifted values
        center = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.grid.n_dims)])
        shifted = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.grid.n_dims)])

        # compute the average
        ave = 0.5*(p_arr[center] + p_arr[shifted])
        return FieldVariable(arr=ave, **self.get_kw())
        

    def diff_forward(self, axis):
        """
        Compute the forward difference in a given direction

        Arguments:
            axis (int): Axis along which to compute the difference

        Returns:
            FieldVariable: Forward difference of the FieldVariable
        """
        pad_width = tuple([(0, 0) if i != axis else (0, 1)
                           for i in range(self.grid.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.grid.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else -1
                        for i in range(self.grid.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.grid.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.grid.n_dims)])

        # calculate the forward difference
        diff = (p_arr[secon] - p_arr[first])/self.grid.dx[axis]
        return FieldVariable(arr=diff, **self.get_kw())
    
    def diff_backward(self, axis):
        """
        Compute the backward difference in a given direction

        Arguments:
            axis (int): Axis along which to compute the difference

        Returns:
            FieldVariable: Backward difference of the FieldVariable
        """
        pad_width = tuple([(0, 0) if i != axis else (1, 0)
                           for i in range(self.grid.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.grid.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else 0
                        for i in range(self.grid.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.grid.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.grid.n_dims)])

        # calculate the backward difference
        diff = (p_arr[secon] - p_arr[first])/self.grid.dx[axis]
        return FieldVariable(arr=diff, **self.get_kw())
        
    
    def diff_2(self, axis):
        """
        Compute the second order difference in a given direction

        Arguments:
            axis (int): Axis along which to compute the difference

        Returns:
            FieldVariable: Second order difference of the FieldVariable
        """
        pad_width = tuple([(0, 0) if i != axis else (1, 1)
                           for i in range(self.grid.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.grid.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else 0
                        for i in range(self.grid.n_dims)])
            p_arr[sl] = 0
            sl = tuple([slice(None) if i != axis else -1
                        for i in range(self.grid.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -2)
                        for i in range(self.grid.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, -1)
                        for i in range(self.grid.n_dims)])
        third = tuple([slice(None) if i != axis else slice(2, None)
                        for i in range(self.grid.n_dims)])

        # calculate the second order difference
        diff = (p_arr[third] - 2*p_arr[secon] + p_arr[first])/self.grid.dx[axis]**2
        return FieldVariable(arr=diff, **self.get_kw())

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key):
        return self.arr[key]
    
    def __setitem__(self, key, value):
        self.arr[key] = value

    def __getattr__(self, name):
        """
        Forward attribute access to the underlying array (e.g. shape)
        """
        try:
            return getattr(self.arr, name)
        except AttributeError:
            raise AttributeError(f"FieldVariable has no attribute {name}")

    # For pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        return FieldVariable(arr=deepcopy(self.arr, memo), 
                             **deepcopy(self.get_kw(), memo))

    # ==================================================================
    #  ARITHMETIC OPERATIONS
    # ==================================================================

    def __add__(self, other):
        """
        Add A FieldVariable to another FieldVariable or a scalar

        # Arguments:
            other        : FieldVariable or array or scalar

        Returns:
            FieldVariable: Sum of self and other (inherits from self)
        """
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            sum = self.arr + other.arr
        else:
            sum = self.arr + other

        return FieldVariable(arr=sum, **kwargs)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract A FieldVariable to another FieldVariable or a scalar

        Arguments:
            other        : FieldVariable or array or scalar

        Returns:
            FieldVariable: Difference of self and other (inherits from self)
        """
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            diff = self.arr - other.arr
        else:
            diff = self.arr - other

        return FieldVariable(arr=diff, **kwargs)
    
    def __rsub__(self, other):
        res = other - self.arr
        return FieldVariable(arr=res, **self.get_kw())

    def __mul__(self, other):
        """
        Multiply A FieldVariable to another FieldVariable or a scalar

        Arguments:
            other        : FieldVariable or array or scalar

        Returns:
            FieldVariable: Product of self and other (inherits from self)
        """
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            prod = self.arr * other.arr
        else:
            prod = self.arr * other

        return FieldVariable(arr=prod, **kwargs)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Divide A FieldVariable to another FieldVariable or a scalar

        Arguments:
            other        : FieldVariable or array or scalar

        Returns:
            FieldVariable: Quotient of self and other (inherits from self)
        """
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            quot = self.arr / other.arr
        else:
            quot = self.arr / other

        return FieldVariable(arr=quot, **kwargs)
    
    def __rtruediv__(self, other):
        return FieldVariable(arr=other / self.arr, **self.get_kw())

    def __pow__(self, other):
        """
        Raise A FieldVariable to another FieldVariable or a scalar

        Arguments:
            other        : FieldVariable or array or scalar

        Returns:
            FieldVariable: Power of self and other (inherits from self)
        """
        kwargs = self.get_kw()
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            topo = [p or q for p, q in zip(self.topo, other.topo)]
            kwargs["topo"] = topo
            pow = self.arr ** other.arr
        else:
            pow = self.arr ** other

        return FieldVariable(arr=pow, **kwargs)

    # ==================================================================
    #  STRING REPRESENTATION
    # ==================================================================

    def __str__(self) -> str:
        return f"FieldVariable: {self.name} \n {self.arr}"

    def __repr__(self) -> str:
        return self.__str__()