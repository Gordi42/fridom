import numpy
try :
    import cupy
except ImportError:
    pass

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.BoundaryConditions import BoundaryConditions


class FieldVariable:
    """
    Base class for FieldVariables

    Attributes:
        name (str)          : Name of the FieldVariable
        mset (ModelSettings): ModelSettings object
        grid (Grid)         : Grid object
        is_spectral (bool)  : True if the FieldVariable is in spectral space
        arr (ndarray)       : The underlying array
        bc (BoundaryCondition) : Boundary condition

    Methods:
        zeros: Create a FieldVariable of zeros
        ones : Create a FieldVariable of ones
    """

    # ==================================================================
    #  CONSTRUCTORS
    # ==================================================================

    def __init__(self, mset:ModelSettingsBase, grid:GridBase,
                 is_spectral=False, name="Unnamed", bc=BoundaryConditions,
                 arr=None) -> None:
        """
        Creates a FieldVariable initialized from input array if given.
        Else, creates a FieldVariable initialized with zeros.

        Arguments:
            input_array (ndarray): The array to be wrapped
            mset (ModelSettings) : ModelSettings object
            grid (Grid)          : Grid object
            is_spectral (bool)   : True if the FieldVariable is in spectral space
            name (str)           : Name of the FieldVariable
            bc (Boundary Condition) : Boundary condition
        """
        self.name = name
        self.mset = mset
        self.grid = grid
        self.is_spectral = is_spectral
        self.bc = bc

        cp = cupy if mset.gpu else numpy
        self.cp = cp
        dtype = mset.ctype if is_spectral else mset.dtype
        if arr is None:
            if is_spectral:
                self.arr = cp.zeros(shape=grid.K[0].shape, dtype=dtype)
            else:
                self.arr = cp.zeros(shape=tuple(mset.N), dtype=dtype)
        else:
            self.arr = cp.array(arr, dtype=dtype)

        self.forward = lambda x: cp.fft.fftn(bc.pad_for_fft(x))
        self.backward = lambda x: bc.unpad_from_fft(cp.fft.ifftn(x).real)
        return

    def copy(self):
        """
        Return a copy of the FieldVariable
        """
        return FieldVariable(arr=self.arr.copy(), **self.get_kw())

    def cpu(self):
        """
        Return a copy of the FieldVariable on the CPU
        """
        mset_cpu = self.mset.copy()
        mset_cpu.gpu = False
        # transform grid to CPU
        grid_cpu = self.grid.cpu

        kw = self.get_kw()
        kw["mset"] = mset_cpu
        kw["grid"] = grid_cpu

        arr = self.arr.get() if self.mset.gpu else self.arr

        return FieldVariable(arr=arr, **kw)
        
    # ==================================================================
    #  OTHER METHODS
    # ==================================================================

    def get_kw(self):
        """
        Return a dictionary with the keyword arguments for the
        FieldVariable constructor
        """
        return {"mset":self.mset, "grid":self.grid, "name":self.name,
                "is_spectral":self.is_spectral, "bc":self.bc}

    def fft(self) -> "FieldVariable":
        """
        Compute forward and backward Fourier transform of the FieldVariable

        Returns:
            FieldVariable: Fourier transform of the FieldVariable
        """
        cp = self.cp
        if self.is_spectral:
            res = cp.array(self.backward(self.arr), dtype=self.mset.dtype)
        else:
            res = cp.array(self.forward(self.arr), dtype=self.mset.ctype)

        kw = self.get_kw()
        kw["is_spectral"] = not self.is_spectral

        return FieldVariable(arr=res, **kw)

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
        cp = self.cp
        if self.is_spectral:
            spectral = self.arr.copy()
        else:
            spectral = self.fft().arr.copy()
        # normalize with respect to the number of grid points
        spectral /= spectral.size
        # check if domain was extended
        fac = numpy.prod([1 if p else 2 for p in self.mset.periodic_bounds])
        spectral /= fac

        # calculate n-dimensional spectra and flatten the array
        spectra_3D = (cp.abs(spectral)**2).flatten()
        # calculate the corresponding wave number
        k = cp.zeros_like(spectral).real
        for K in self.grid.K:
            k += K**2
        k = cp.sqrt(k).flatten()

        # find largest grid spacing in spectral space
        dk_max = 0
        k_vol = 1
        for kk in self.grid.k:
            dk_max = max(dk_max, cp.abs(kk[1] - kk[0]))
            k_vol *= kk[1] - kk[0]

        kmax = float(cp.max(k))
        dk_max = float(dk_max)
        # find the bin edges
        k_bins = cp.arange(0, kmax + dk_max, dk_max)
        # find the bin centers
        k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
        # integrate the 3D spectra over the bins
        k_idx = cp.digitize(k, k_bins)
        spectra_1D = cp.bincount(k_idx, weights=spectra_3D)[1:]
        # normalize:
        spectra_1D *= k_vol/(k_centers[1] - k_centers[0])

        return k_centers, spectra_1D

    def sqrt(self):
        """
        Compute the square root of the FieldVariable

        Returns:
            FieldVariable: Square root of the FieldVariable
        """
        return FieldVariable(arr=self.cp.sqrt(self.arr), **self.get_kw())

    def norm_l2(self):
        """
        Compute the L2 norm of the FieldVariable

        Returns:
            float: L2 norm of the FieldVariable
        """
        return self.cp.linalg.norm(self.arr)

    def pad_raw(self, pad_width):
        """
        Return a padded version of the FieldVariable with the given padding
        width. Padded values are periodic.

        Arguments:
            pad_width (int or tuple): Padding width

        Returns:
            p_arr (ndarray): Padded version of the FieldVariable
        """
        cp = self.cp
        p_arr = cp.pad(self.arr, pad_width, mode="wrap")
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
                               for i in range(self.mset.n_dims)])
        else:
            pad_width = tuple([(0, 0) if i != axis else (1, 0) 
                               for i in range(self.mset.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.mset.periodic_bounds[axis]:
            if shift_positive:
                sl = tuple([slice(None) if i != axis else -1
                            for i in range(self.mset.n_dims)])
            else:
                sl = tuple([slice(None) if i != axis else 0
                            for i in range(self.mset.n_dims)])
            p_arr[sl] = 0
                

        # get slices for the center and shifted values
        center = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.mset.n_dims)])
        shifted = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.mset.n_dims)])

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
                           for i in range(self.mset.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.mset.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else -1
                        for i in range(self.mset.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.mset.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.mset.n_dims)])

        # calculate the forward difference
        diff = (p_arr[secon] - p_arr[first])/self.mset.dg[axis]
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
                           for i in range(self.mset.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.mset.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else 0
                        for i in range(self.mset.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -1)
                        for i in range(self.mset.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, None)
                         for i in range(self.mset.n_dims)])

        # calculate the backward difference
        diff = (p_arr[secon] - p_arr[first])/self.mset.dg[axis]
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
                           for i in range(self.mset.n_dims)])

        # pad the array
        p_arr = self.pad_raw(pad_width)

        # apply boundary conditions
        if not self.mset.periodic_bounds[axis]:
            sl = tuple([slice(None) if i != axis else 0
                        for i in range(self.mset.n_dims)])
            p_arr[sl] = 0
            sl = tuple([slice(None) if i != axis else -1
                        for i in range(self.mset.n_dims)])
            p_arr[sl] = 0

        # get slices for the center and shifted values
        first = tuple([slice(None) if i != axis else slice(None, -2)
                        for i in range(self.mset.n_dims)])
        secon = tuple([slice(None) if i != axis else slice(1, -1)
                        for i in range(self.mset.n_dims)])
        third = tuple([slice(None) if i != axis else slice(2, None)
                        for i in range(self.mset.n_dims)])

        # calculate the second order difference
        diff = (p_arr[third] - 2*p_arr[secon] + p_arr[first])/self.mset.dg[axis]**2
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
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            sum = self.arr + other.arr
        else:
            sum = self.arr + other

        return FieldVariable(arr=sum, **self.get_kw())
    
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
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            diff = self.arr - other.arr
        else:
            diff = self.arr - other

        return FieldVariable(arr=diff, **self.get_kw())
    
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
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            prod = self.arr * other.arr
        else:
            prod = self.arr * other

        return FieldVariable(arr=prod, **self.get_kw())
    
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
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            quot = self.arr / other.arr
        else:
            quot = self.arr / other

        return FieldVariable(arr=quot, **self.get_kw())
    
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
        # Check that the other object is a FieldVariable
        if isinstance(other, FieldVariable):
            pow = self.arr ** other.arr
        else:
            pow = self.arr ** other

        return FieldVariable(arr=pow, **self.get_kw())

    # ==================================================================
    #  STRING REPRESENTATION
    # ==================================================================

    def __str__(self) -> str:
        return f"FieldVariable: {self.name} \n {self.arr}"

    def __repr__(self) -> str:
        return self.__str__()
