# Import external modules
from typing import TYPE_CHECKING
from copy import deepcopy
from functools import partial
# Import internal modules
from fridom.framework import config, utils
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.grid.position_base import PositionBase


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
    _dynamic_attributes = ["mset", "arr"]
    def __init__(self, 
                 mset: 'ModelSettingsBase',
                 name: str,
                 position: 'PositionBase',
                 is_spectral=False, 
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
        self.position = position
        self.long_name = long_name
        self.units = units
        self.nc_attrs = nc_attrs or {}
        self.mset = mset
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
                "position": self.position,
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
        from copy import copy
        f = copy(self)
        f.arr = res
        f.is_spectral = not self.is_spectral

        return f

    def sync(self) -> 'FieldVariable':
        """
        Synchronize the FieldVariable (exchange boundary values)
        """
        f = self
        f.arr = self.grid.sync(self.arr)
        return f

    def apply_boundary_conditions(self, 
                                  axis: int, 
                                  side: str, 
                                  value: 'float | np.ndarray | FieldVariable'
                                  ) -> 'FieldVariable':
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
        f = self
        f.arr = self.grid.apply_boundary_condition(self.arr, axis, side, value)
        return f

    def norm_l2(self) -> float:
        """
        Compute the L2 norm of the FieldVariable

        Returns:
            float: L2 norm of the FieldVariable
        """
        return config.ncp.linalg.norm(self.arr)

    # ==================================================================
    #  SLICING
    # ==================================================================

    def __getitem__(self, key):
        return self.arr[key]
    
    def __setitem__(self, key, value):
        new_arr = utils.modify_array(self.arr, key, value)
        self.arr = new_arr

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

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def grid(self):
        """Return the grid of the FieldVariable"""
        return self.mset.grid


utils.jaxify_class(FieldVariable)