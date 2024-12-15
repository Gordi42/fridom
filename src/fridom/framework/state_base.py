"""Base class for the state vector of a model."""
from __future__ import annotations

# Import external modules
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Callable

# Import internal modules
import fridom.framework as fr

# Import type information
if TYPE_CHECKING:
    import numpy as np
    import xarray


@partial(fr.utils.jaxify, dynamic=("fields",))
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
    mset : fr.ModelSettingsBase
        The model settings
    field_list : list[FieldVariable] | dict[str, FieldVariable]
        The list of fields that make up the state.
    is_spectral : bool
        Whether the state is in spectral space. (default: False)

    """

    # ======================================================================
    #  STATE CONSTRUCTORS
    # ======================================================================

    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 field_list: list | dict | None = None,
                 is_spectral: bool = False) -> None:  # noqa: FBT001, FBT002

        _ = is_spectral  # unused
        self.mset = mset
        if type(field_list) is list:
            self.fields = {field.name: field for field in field_list}
        elif type(field_list) is dict:
            self.fields = field_list
        elif field_list is None:
            self.fields = {}
        else:
            msg = "field_list must be a list or a dictionary."
            raise TypeError(msg)

    # ======================================================================
    #  BASIC OPERATIONS
    # ======================================================================

    def fft(self,
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            ) -> StateBase:
        """
        Calculate the Fourier transform of the state (forward and backward).

        Parameters
        ----------
        padding : fr.grid.FFTPadding
            Zero padding type for the FFT (default is no padding)

        Returns
        -------
        StateBase
            The Fourier transformed state

        """
        if self.is_spectral:
            msg = "State is in spectral space, cannot perform fft"
            raise ValueError(msg)
        # loop over all fields in self.field_dict
        transformed_fields = [field.fft(padding) for field in self.fields.values()]
        return self.__class__(self.mset, field_list=transformed_fields)

    def ifft(self,
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             ) -> StateBase:
        """
        Calculate the inverse Fourier transform of the state.

        Parameters
        ----------
        padding : fr.grid.FFTPadding
            Zero padding type for the FFT (default is no padding)

        Returns
        -------
        StateBase
            The inverse Fourier transformed state

        """
        if not self.is_spectral:
            msg = "State is not in spectral space, cannot perform ifft"
            raise ValueError(msg)
        # loop over all fields in self.field_dict
        transformed_fields = [field.ifft(padding) for field in self.fields.values()]
        return self.__class__(self.mset, field_list=transformed_fields)

    def sync(self) -> StateBase:
        """Synchronize the state (exchange ghost cells)."""
        # sync all the arrays
        arrs = self.grid.sync_multi(tuple(field.arr for field in self.fields.values()))
        # set the arrays to the fields
        for field, arr in zip(self.fields.values(), arrs):
            field.arr = arr
        # apply boundary conditions to the fields
        for field in self.fields.values():
            field.apply_water_mask()
        return self

    def project(self,
                p_vec: StateBase,
                q_vec: StateBase) -> StateBase:
        r"""
        Project the state on a (spectral) vector.

        Description
        -----------
        The projection of the state :math:`\boldsymbol{z}` on a P-Vector
        :math:`\boldsymbol{p}` and a Q-Vector :math:`\boldsymbol{q}` is defined as:

        .. math::
            \boldsymbol{z} = \boldsymbol{q} \cdot \left(
                \boldsymbol{z} \cdot \boldsymbol{p}
            \right)

        The projection is done in spectral space. All states are transformed to
        spectral space before the projection and transformed back to physical space
        if necessary.

        Parameters
        ----------
        p_vec : StateBase
            the projection vector :math:`\boldsymbol{p}`
        q_vec : StateBase
            the polarization vector :math:`\boldsymbol{q}`

        Returns
        -------
        StateBase
            The projected state :math:`\boldsymbol{z}`

        """
        # transform to spectral space if necessary
        was_spectral = self.is_spectral
        z = self if was_spectral else self.fft()
        # check if the projection vectors are in spectral space
        if not p_vec.is_spectral:
            p_vec = p_vec.fft()
        if not q_vec.is_spectral:
            q_vec = q_vec.fft()
        # project
        z = q_vec * (z.dot(p_vec))
        # transform back to physical space if necessary
        if not was_spectral:
            z = z.ifft()
        return z

    def dot(self, other: StateBase) -> fr.FieldVariable:
        r"""
        Calculate the dot product of the state with another state.

        Description
        -----------
        The dot product of two states :math:`\boldsymbol{z}` and
        :math:`\boldsymbol{z}'` is defined as:

        .. math::
            \boldsymbol{z} \cdot \boldsymbol{z}'
                = \sum_i \boldsymbol{z}_i \overline{\boldsymbol{z}'}_i

        where :math:`\boldsymbol{z}_i` are the fields of the state.

        Parameters
        ----------
        other : StateBase
            The other state

        Returns
        -------
        FieldVariable
            The dot product of the two states

        """
        return sum(self.fields[key] * other.fields[key].arr.conj()
                   for key in self.fields)

    def norm_l2(self) -> float:
        r"""
        Calculate the L2 norm of the state.

        Description
        -----------
        The L2 norm of the state :math:`\boldsymbol{z}` is defined as:

        .. math::
            ||\boldsymbol{z}||_2 = \sqrt{
                \int \boldsymbol{z} \cdot \boldsymbol{z} \, dV

        in practice, the integral is calculated as a sum over the grid cells
        and :math:`dV` is the cell volume of each grid cell.

        Returns
        -------
        float
            The L2 norm of the state

        """
        ncp = fr.config.ncp
        cell_volume = self.grid.dV
        return ncp.sqrt(ncp.sum(self.dot(self).unpad()) * cell_volume)

    def norm_of_diff(self, other: StateBase) -> float:
        r"""
        Norm of difference between two states.

        Description
        -----------
        The norm of difference computes the normalized difference between
        two states :math:`\boldsymbol{z}` and :math:`\boldsymbol{z}'`.
        It is defined as:

        .. math::
            2 \frac{||\boldsymbol{z} - \boldsymbol{z}'||}
                   {||\boldsymbol{z}|| + ||\boldsymbol{z}'||}

        where :math:`||\cdot||` is the L2 norm of the state. The norm of
        difference is in the range [0, 2].

        Parameters
        ----------
        other : StateBase
            The other state to compare with

        Returns
        -------
        float
            The norm of difference between the two states

        """
        return 2 * (self - other).norm_l2() / (self.norm_l2() + other.norm_l2())

    def has_nan(self) -> bool:
        """Check if the state contains NaN values."""
        return any(field.has_nan() for field in self.fields.values())

    def __repr__(self) -> str:
        res = "State with fields:\n"
        for field in self.fields.values():
            res += f"  {field.name}: {field.long_name}  [{field.units}]\n"
        return res

    # ================================================================
    #  xarray conversion
    # ================================================================
    @property
    def xr(self) -> xarray.Dataset:
        """Convert the state to an xarray dataset."""
        return self.xrs[:]

    @property
    def xrs(self) -> fr.utils.SliceableAttribute[xarray.Dataset]:
        """
        Convert a slice of the state to an xarray dataset.

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The slice of the state to convert

        Example
        -------
        Let's say we have a 3D state `z`. We can convert
        the top layer of the state to an xarray dataset as follows:

        .. code-block:: python

            data_set = z.xrs[:,:,-1]  # Only the top layer

        """
        import xarray as xr
        def slicer(key: int | slice | tuple[int | slice]) -> xr.Dataset:
            return xr.Dataset(
                {field.name: field.xrs[key] for field in self.fields.values()})
        return fr.utils.SliceableAttribute(slicer)

    @classmethod
    def from_xarray(cls,
                    mset: fr.ModelSettingsBase,
                    ds: xarray.Dataset) -> StateBase:
        """
        Create a state from an xarray dataset.

        Parameters
        ----------
        mset : fr.ModelSettingsBase
            The model settings
        ds : xarray.Dataset
            The xarray dataset to convert

        Returns
        -------
        StateBase
            The state created from the xarray dataset

        """
        # get the list of variable names
        var_names = list(ds.variables)
        # remove the dimensions from the list
        var_names = list(set(var_names) - set(ds.dims))
        # create the field list
        field_list = [fr.FieldVariable.from_xarray(mset, ds[var_name])
                      for var_name in var_names]
        # create the state
        return cls(mset, field_list=field_list)

    def to_netcdf(self, path: str) -> None:
        """
        Write the state to a NetCDF file.

        Parameters
        ----------
        path : str
            The path to the NetCDF file

        """
        self.xr.to_netcdf(path, auto_complex=True)

    @classmethod
    def from_netcdf(cls,
                    mset: fr.ModelSettingsBase,
                    path: str) -> None:
        """
        Read the state from a NetCDF file.

        Parameters
        ----------
        mset : fr.ModelSettingsBase
            The model settings
        path : str
            The path to the NetCDF file

        Returns
        -------
        StateBase
            The state read from the NetCDF file

        """
        import xarray as xr
        ds = xr.open_dataset(path)
        return cls.from_xarray(mset, ds)

    # ================================================================
    #  FieldVariable access
    # ================================================================
    def __getitem__(self, key: str) -> fr.FieldVariable:
        """Access the state by field name."""
        return self.fields[key]

    def __setitem__(self, key: str, value: fr.FieldVariable) -> None:
        """Set the state by field name."""
        # check if the field is in the state
        if key not in self.fields:
            msg = f"Field {key} not in state"
            raise KeyError(msg)
        self.fields[key] = value

    # ================================================================
    #  PROPERTIES
    # ================================================================

    @property
    def field_list(self) -> list:
        """Return the list of fields."""
        return list(self.fields.values())

    @property
    def arr_dict(self) -> dict[str, np.ndarray]:
        """Return the dictionary of arrays (not FieldVariables)."""
        return {field.name: field.arr for field in self.fields.values()}

    @arr_dict.setter
    def arr_dict(self, arr_dict: dict[str, np.ndarray]) -> None:
        """Set the dictionary of arrays (not FieldVariables)."""
        for key, field in self.fields.items():
            field.arr = arr_dict[key]

    @property
    def grid(self) -> fr.grid.GridBase:
        """Return the grid of the model."""
        return self.mset.grid

    @property
    def is_spectral(self) -> bool:
        """Return whether the state is in spectral space."""
        if len(self.fields) == 0:
            msg = "State has no fields, cannot determine if spectral."
            msg += " Returning False."
            fr.log.warning(msg)
            return False
        # get the first field and return its is_spectral property
        return self.field_list[0].is_spectral

    # ================================================================
    #  Creating copies
    # ================================================================
    def __copy__(self) -> StateBase:
        """Create a copy of the fields but not of the model settings."""
        field_list = [copy(field) for field in self.field_list]
        return self.__class__(
            mset=self.mset,
            field_list=field_list,
            is_spectral=self.is_spectral)

    # ----------------------------------------------------------------
    #  NetCDF I/O
    # ----------------------------------------------------------------

    # ======================================================================
    #  OPERATOR OVERLOADING
    # ======================================================================

    @staticmethod
    def _apply_operation(op: Callable[[StateBase, any], StateBase],
                         state: StateBase,
                         other: any) -> StateBase:
        cls = state.__class__
        keys = state.fields.keys()
        if isinstance(other, cls):
            res = {key: op(state.fields[key], other.fields[key]) for key in keys}
        else:
            res = {key: op(state.fields[key], other) for key in keys}

        return cls(state.mset, field_list=res, is_spectral=state.is_spectral)

    def __add__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: x + y, self, other)

    def __radd__(self, other: StateBase | any) -> StateBase:
        return self.__add__(other)

    def __sub__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: x - y, self, other)

    def __rsub__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: y - x, self, other)

    def __mul__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: x * y, self, other)

    def __rmul__(self, other: StateBase | any) -> StateBase:
        return self.__mul__(other)

    def __truediv__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: x / y, self, other)

    def __rtruediv__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: y / x, self, other)

    def __pow__(self, other: StateBase | any) -> StateBase:
        return self._apply_operation(lambda x, y: x ** y, self, other)

    def __matmul__(self, other: StateBase) -> StateBase:
        """Dot product of two states."""
        return self.dot(other)
