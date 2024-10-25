# Import external modules
from typing import TYPE_CHECKING
from functools import partial
from copy import copy
# Import internal modules
import fridom.framework as fr
from fridom.framework import config, utils
from fridom.framework.grid.fft_padding import FFTPadding
# Import type information
if TYPE_CHECKING:
    import numpy as np
    from fridom.framework.field_variable import FieldVariable
    from fridom.framework.model_settings_base import ModelSettingsBase

@partial(utils.jaxify, dynamic=("fields",))
class StateBase:
    """
    Base class for a model state.
    
    Description
    -----------
    A model state is a collection of fields that represent the state of the model.
    This class provides basic operations on the state, such as addition, subtraction,
    multiplication, division, as well as dot products, norms, and fourier transforms.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings
    `field_list` : `list[FieldVariable]` | `dict[str, FieldVariable]`
        The list of fields that make up the state.
    `is_spectral` : `bool`
        Whether the state is in spectral space. (default: False)
    
    Attributes
    ----------
    `fields` : `dict[str, FieldVariable]`
        A dictionary of fields that make up the state.
    
    Methods
    -------
    `fft()` -> `State`
        Calculate the Fourier transform of the state.
    `sync()` -> `None`
        Synchronize the state. (Exchange ghost cells)
    `project(p_vec: State, q_vec: State)` -> `State`
        Project the state on a (spectral) vector.
    `dot(other: State)` -> `FieldVariable`
        Calculate the dot product of the state with another state.
    `norm_l2()` -> `float`
        Calculate the L2 norm of the state.
    `norm_of_diff(other: State)` -> `float`
        Calculate the norm of the difference between two states.
    """
    # ======================================================================
    #  STATE CONSTRUCTORS
    # ======================================================================

    def __init__(self, mset: 'ModelSettingsBase', 
                 field_list: list | dict, is_spectral=False) -> None:
        
        self.mset = mset
        self.is_spectral = is_spectral
        if type(field_list) is list:
            self.fields = {field.name: field for field in field_list}
        elif type(field_list) is dict:
            self.fields = field_list
        else:
            raise TypeError("field_list must be a list or a dictionary.")
        return
    
    # ======================================================================
    #  BASIC OPERATIONS
    # ======================================================================
        
    def fft(self,
            padding = FFTPadding.NOPADDING) -> "StateBase":
        """
        Calculate the Fourier transform of the state. (forward and backward)
        """
        # loop over all fields in self.field_dict
        fields_fft = [field.fft(padding) for field in self.fields.values()]
        z = self.__class__(
            self.mset, field_list=fields_fft, 
            is_spectral=not self.is_spectral)
        return z

    def ifft(self,
                padding = FFTPadding.NOPADDING) -> "StateBase":
        """
        Calculate the inverse Fourier transform of the state.
        """
        if not self.is_spectral:
            raise ValueError("FieldVariable is not in spectral space, cannot perform ifft")
        return self.fft(padding=padding)

    def sync(self) -> None:
        """
        Synchronize the state. (Exchange ghost cells)
        """
        arrs = self.grid.sync_multi(tuple(field.arr for field in self.fields.values()))
        for field, arr in zip(self.fields.values(), arrs):
            field.arr = arr
        return

    def project(self, p_vec:"StateBase", 
                      q_vec:"StateBase") -> "StateBase":
        """
        Project the state on a (spectral) vector.
        $ z = q_vec * (z \\cdot p_vec) $

        Arguments:
            p_vec (State)  : P-Vector.
            q_vec (State)  : Q-Vector.
        """
        # transform to spectral space if necessary
        was_spectral = self.is_spectral
        if was_spectral:
            z = self
        else:
            z = self.fft()
        # project
        z = q_vec * (z.dot(p_vec))
        # transform back to physical space if necessary
        if not was_spectral:
            z = z.fft()
        return z

    def dot(self, other: "StateBase") -> 'FieldVariable':
        """
        Calculate the dot product of the state with another state.

        Arguments:
            other (State)  : Other state (gets complex conjugated).
        """
        return sum(self.fields[key] * other.fields[key].arr.conj() 
                   for key in self.fields.keys())

    def norm_l2(self) -> float:
        """
        Calculate the L2 norm of the state.

        $$ ||z||_2 = \\sqrt{ \\sum_{i} \\int z_i^2 dV } $$
        where $z_i$ are the fields of the state.

        Returns:
            norm (float)  : L2 norm of the state.
        """
        ncp = config.ncp
        cell_volume = self.grid.dV
        return ncp.sqrt(ncp.sum(self.dot(self).unpad()) * cell_volume)

    def norm_of_diff(self, other: "StateBase") -> float:
        r"""
        The norm of the difference between two states.

        .. math::
            2 \frac{||z - z'||_2}{||z||_2 + ||z'||_2}
        """
        return 2 * (self - other).norm_l2() / (self.norm_l2() + other.norm_l2())

    def has_nan(self) -> bool:
        """
        Check if the state contains NaN values.
        """
        return any(field.has_nan() for field in self.fields.values())

    # ================================================================
    #  xarray conversion
    # ================================================================
    @property
    def xr(self):
        """
        State as xarray dataset
        """
        return self.xrs[:]

    @property
    def xrs(self):
        """
        State of sliced domain as xarray dataset 
        """
        import xarray as xr
        def slicer(key):
            dvs = {field.name: field.xrs[key] for field in self.fields.values()}
            return xr.Dataset(dvs)
        return fr.utils.SliceableAttribute(slicer)

    # ================================================================
    #  FieldVariable access
    # ================================================================
    def __getitem__(self, key):
        """
        Access the state by field name.
        """
        return self.fields[key]

    def __setitem__(self, key, value):
        """
        Set the state by field name.
        """
        self.fields[key] = value
        return

    # ================================================================
    #  PROPERTIES
    # ================================================================

    def __repr__(self):
        """
        Return the string representation of the state.
        """
        res = "State with fields:\n"
        for field in self.fields.values():
            res += f"  {field.name}: {field.long_name}  [{field.units}]\n"
        return res

    @property
    def field_list(self) -> list:
        """
        Return the list of fields.
        """
        return list(self.fields.values())

    @property
    def arr_dict(self) -> 'dict[str, np.ndarray]':
        """
        Return the dictionary of arrays (not FieldVariables).
        """
        return {field.name: field.arr for field in self.fields.values()}
    @arr_dict.setter
    def arr_dict(self, arr_dict: 'dict[str, np.ndarray]') -> None:
        """
        Set the dictionary of arrays (not FieldVariables).
        """
        for key, field in self.fields.items():
            field.arr = arr_dict[key]
        return

    # ================================================================
    #  Creating copies
    # ================================================================
    def __copy__(self):
        field_list = [copy(field) for field in self.field_list]
        return self.__class__(
            mset=self.mset, 
            field_list=field_list, 
            is_spectral=self.is_spectral)

    # ----------------------------------------------------------------
    #  NetCDF I/O
    # ----------------------------------------------------------------
    def to_netcdf(self, path: str) -> None:
        """
        Write the state to a NetCDF file.
        """
        self.xr.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, 
                    mset: 'ModelSettingsBase',
                    path: str) -> None:
        """
        Read the state from a NetCDF file.
        """
        import xarray as xr
        ncp = fr.config.ncp

        ds = xr.open_dataset(path)
        z = cls(mset)
        for key in z.fields.keys():
            z.fields[key].arr = mset.grid.pad(ncp.array(ds[key]).T)
        z.sync()
        return z

    # ======================================================================
    #  OPERATOR OVERLOADING
    # ======================================================================

    def __add__(self, other):
        """
        Add two states / fields together.
        """
        keys = self.fields.keys()
        if isinstance(other, self.__class__):
            sums = {key: self.fields[key] + other.fields[key] for key in keys}
        else:
            sums = {key: self.fields[key] + other for key in keys}

        z = self.__class__(self.mset, field_list=sums,
                                is_spectral=self.is_spectral)
        
        return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two states / fields.
        """
        keys = self.fields.keys()
        if isinstance(other, self.__class__):
            diffs = {key: self.fields[key] - other.fields[key] for key in keys}
        else:
            diffs = {key: self.fields[key] - other for key in keys}

        z = self.__class__(self.mset, field_list=diffs,
                                is_spectral=self.is_spectral)
        return z
    
    def __rsub__(self, other):
        """
        Subtract something from the state.
        """
        diffs = [other - field for field in self.field_list]
        z = self.__class__(self.mset, field_list=diffs,
                                is_spectral=self.is_spectral)
        return z
        
    def __mul__(self, other):
        """
        Multiply two states / fields.
        """
        keys = self.fields.keys()
        if isinstance(other, self.__class__):
            prods = {key: self.fields[key] * other.fields[key] for key in keys}
        else:
            prods = {key: self.fields[key] * other for key in keys}

        z = self.__class__(self.mset, field_list=prods,
                                is_spectral=self.is_spectral)
        return z
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divide two states / fields.
        """
        keys = self.fields.keys()
        if isinstance(other, self.__class__):
            quot = {key: self.fields[key] / other.fields[key] for key in keys}
        else:
            quot = {key: self.fields[key] / other for key in keys}

        z = self.__class__(self.mset, field_list=quot,
                                is_spectral=self.is_spectral)
        return z

    def __rtruediv__(self, other):
        """
        Divide something by the state.
        """
        keys = self.fields.keys()
        quot = {key: other / self.fields[key] for key in keys}
        z = self.__class__(self.mset, field_list=quot,
                                is_spectral=self.is_spectral)
        return z
    
    def __pow__(self, other):
        """
        Exponentiate two states / fields.
        """
        keys = self.fields.keys()
        if isinstance(other, self.__class__):
            prods = {key: self.fields[key] ** other.fields[key] for key in keys}
        else:
            prods = {key: self.fields[key] ** other for key in keys}

        z = self.__class__(self.mset, field_list=prods,
                                is_spectral=self.is_spectral)
        return z

    def __matmul__(self, other):
        """
        Dot product of two states.
        """
        return self.dot(other)

    @property
    def grid(self):
        """
        Return the grid of the model.
        """
        return self.mset.grid
