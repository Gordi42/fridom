"""The vector field module."""
from __future__ import annotations

from collections import OrderedDict
from copy import copy
from functools import partial
from typing import Callable, Iterator, TypeVar

import numpy as np

import fridom.framework as fr

T = TypeVar("T", bound="VectorField")

@partial(fr.utils.jaxify, dynamic=("_fields", "_vector_dim"))
class VectorField(fr.FieldBase):

    """
    A vector mapping from grid space to the vector space.

    Description
    -----------
    A vector field essentially is a list of scalar fields, one for each
    component of the vector field. This class provides additional methods
    for vector operations such as dot product, or differential operators like
    divergence.

    Parameters
    ----------
    mset : fr.ModelSettingsBase
        The model settings object.
    field_list : list[fr.ScalarField] | OrderedDict[str, fr.ScalarField] | None
        The list of scalar fields that make up the vector field.
    vector_dim : int | None
        The vector dimension. If None, it is set to the length of field_list.
    **kwargs : any
        Additional keyword arguments to pass to the scalar fields.

    """

    # ================================================================
    #  Constructor
    # ================================================================

    def __init__(self,
                 mset: fr.ModelSettingsBase,
                 field_list: (list[fr.ScalarField] |
                              OrderedDict[str, fr.ScalarField] |
                              None) = None,
                 vector_dim: int | None = None,
                 **kwargs: any,
                 ) -> None:
        super().__init__(mset)

        if isinstance(field_list, (list, OrderedDict)) and kwargs:
            msg = "Keyword arguments not allowed when passing a list or dict"
            raise TypeError(msg)

        if kwargs:
            self._check_for_valid_kwargs(kwargs)

        # if the input is a list, check for duplicated names and convert to dict
        if isinstance(field_list, list):
            field_names = [field.name for field in field_list]
            if len(field_names) != len(set(field_names)):
                msg = f"Duplicated field names: {field_names}"
                raise ValueError(msg)
            field_list = OrderedDict((field.name, field) for field in field_list)
        elif isinstance(field_list, OrderedDict):
            pass
        elif field_list is None:
            field_list = self._create_default_fields(mset, vector_dim, **kwargs)
        else:
            msg = f"Invalid field list type: {type(field_list)}"
            raise TypeError(msg)

        # check the vector dimension
        vector_dim = vector_dim or len(field_list)
        if vector_dim != len(field_list):
            msg = f"Vector dimension mismatch: {vector_dim} != {len(field_list)}"
            raise ValueError(msg)

        # set the properties
        self._fields = field_list
        self._vector_dim = vector_dim

    def _check_for_valid_kwargs(self, kwargs: any) -> None:
        allowed_keys = {
            "topo", "is_spectral", "position", "bc_types", "flags", "nc_attrs"}
        if not set(kwargs).issubset(allowed_keys):
            msg = f"Invalid keyword arguments: {set(kwargs) - allowed_keys}"
            raise TypeError(msg)


    @staticmethod
    def _create_default_fields(mset: fr.ModelSettingsBase,
                               vector_dim: int,
                               **kwargs: any,
                               ) -> OrderedDict[str, fr.ScalarField]:
        # check the vector dimension
        if vector_dim is None or vector_dim < 1:
            msg = f"Invalid vector dimension: {vector_dim}"
            raise ValueError(msg)
        # create a dictionary of scalar fields
        field_list = OrderedDict()
        for i in range(vector_dim):
            field = fr.ScalarField(mset, name=f"f{i}", **kwargs)
            field_list[field.name] = field
        return field_list

    # ================================================================
    #  General Methods
    # ================================================================

    def fft(self: T,  # noqa: D102
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            ) -> T:
        return self.apply_elementwise(self,
                                      lambda field: field.fft(padding=padding))

    def ifft(self: T,  # noqa: D102
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             ) -> T:
        return self.apply_elementwise(self,
                                      lambda field: field.ifft(padding=padding))

    def project(self: T,
                p_vec: T,
                q_vec: T) -> T:
        r"""
        Project a Vector Field onto a (spectral) vector.

        Description
        -----------
        The projection of the vector :math:`\boldsymbol{z}` on a P-Vector
        :math:`\boldsymbol{z}` and a Q-Vector :math:`\boldsymbol{q}` is defined as:

        .. math::
            \boldsymbol{z} = \boldsymbol{q} \cdot \left(
                \boldsymbol{z} \cdot \boldsymbol{p}
            \right)

        The projection is done in spectral space. All vectors are transformed to
        spectral space before the projection and transformed back to physical space
        if necessary.

        Parameters
        ----------
        p_vec : VectorField
            the projection vector :math:`\boldsymbol{p}`
        q_vec : VectorField
            the polarization vector :math:`\boldsymbol{q}`

        Returns
        -------
        VectorField
            The projected vector field :math:`\boldsymbol{z}`

        """
        # transform to spectral space if necessary
        was_spectral = self.is_spectral
        vec = self if was_spectral else self.fft()
        # check if the projection vectors are in spectral space
        if not p_vec.is_spectral:
            p_vec = p_vec.fft()
        if not q_vec.is_spectral:
            q_vec = q_vec.fft()
        # project
        vec = q_vec * (vec.dot(p_vec))
        # transform back to physical space if necessary
        if not was_spectral:
            vec = vec.ifft()
        return vec

    def sync(self: T) -> T:  # noqa: D102
        if self.is_spectral:
            # nothing to synchronize in spectral space
            return self
        # sync all arrays at once
        arrs = [field.arr for field in self.fields.values()]
        arrs = self.grid.sync_multi(arrs)
        # set the arrays to the fields
        for field, arr in zip(self.fields.values(), arrs):
            field.arr = arr
        # apply the water mask
        return self.apply_water_mask()

    def apply_water_mask(self: T) -> T:  # noqa: D102
        if self.is_spectral:
            msg = "Vector Field is in spectral space, cannot apply water mask"
            raise ValueError(msg)
        for field in self.fields.values():
            field.apply_water_mask()
        return self

    def has_nan(self) -> bool:  # noqa: D102
        return any(field.has_nan() for field in self.fields.values())

    def __copy__(self: T) -> T:
        # create a new vector field, but copy the fields
        return self.apply_elementwise(self, lambda field: copy(field))

    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self: T,  # noqa: D102
             axis: int,
             order: int = 1,
             ) -> T:
        return self.apply_elementwise(self, lambda field: field.diff(axis, order))

    def grad(self,  # noqa: D102
             axes: list[int] | None = None,
             ) -> fr.TensorField:
        # TODO(Silvano): add implementation after TensorField is implemented
        pass

    def laplacian(self: T,  # noqa: D102
                  axes: tuple[int] | None = None,
                  ) -> T:
        return self.apply_elementwise(self, lambda field: field.laplacian(axes))

    def div(self) -> fr.ScalarField:  # noqa: D102
        return self.grid.diff_module.div(vec=self)

    # ================================================================
    #  xarray Interface
    # ================================================================

    @property
    def xr(self) -> xr.Dataset:  # noqa: D102
        return self.xrs[:]

    @property
    def xrs(self) -> fr.utils.SliceableAttribute[xr.Dataset]:  # noqa: D102
        import xarray as xr
        def slicer(key: int | slice | tuple[int | slice]) -> xr.Dataset:
            ds = xr.Dataset(
                {field.name: field.xrs[key] for field in self.fields.values()})
            # we need to add the variable names in the correct order
            # this ensures that the order of the variables is preserved
            # when loading the vector field from xarray
            var_names = list(ds.variables)
            ds.attrs["var_names"] = var_names
            ds.attrs["vector_dim"] = self.vector_dim
            return ds
        return fr.utils.SliceableAttribute(slicer)

    @classmethod
    def from_xarray(cls: type[T],  # noqa: D102
                    mset: fr.ModelSettingsBase,
                    ds: xr.Dataset,
                    ) -> T:
        # get the list of variable names
        var_names = ds.attrs["var_names"]
        vector_dim = ds.attrs["vector_dim"]
        # create the field list
        field_list = [fr.FieldVariable.from_xarray(mset, ds[var_name])
                      for var_name in var_names]
        # create the vector field
        return cls(mset,
                   field_list=field_list,
                   vector_dim=vector_dim)

    # ================================================================
    #  Sliceable Interface
    # ================================================================

    def __getitem__(self,
                    key: str | int | slice[int]) -> fr.ScalarField | fr.VectorField:
        """Get a field or slice of the vector field."""
        if isinstance(key, str):
            return self.fields[key]
        if isinstance(key, int):
            # get the name of the field at the index
            name = list(self.fields)[key]
            return self.fields[name]
        if isinstance(key, slice):
            # get the slice of the fields
            field_list = list(self.fields.values())[key]
            return VectorField(self.mset,
                               field_list=field_list,
                               vector_dim=len(field_list))
        msg = f"Invalid key type: {type(key)}"
        raise ValueError(msg)

    def __setitem__(self,
                    key: str | int | slice[int],
                    value: fr.ScalarField | fr.VectorField) -> None:
        """Set a field or slice of the vector field."""
        if isinstance(key, str):
            self._check_for_name_mismatch(key, value)
            self.fields[key] = value
            return
        if isinstance(key, int):
            # get the name of the field at the index
            name = list(self.fields)[key]
            self._check_for_name_mismatch(name, value)
            self.fields[name] = value
            return
        if isinstance(key, slice):
            # get the names of the fields in the slice
            names = list(self.fields)[key]
            for name, field in zip(names, value.fields.values()):
                self._check_for_name_mismatch(name, field)
                # set the fields in the slice
                self.fields[name] = field
            return
        msg = f"Invalid key type: {type(key)}"
        raise TypeError(msg)

    def __iter__(self) -> Iterator[fr.ScalarField]:
        """Iterate over the fields of the vector field."""
        return iter(self.fields.values())

    def _check_for_name_mismatch(self, name: str, field: fr.ScalarField) -> None:
        if field.name != name:
            msg = f"Field name mismatch: {field.name} != {name}"
            raise ValueError(msg)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def info(self) -> dict:  # noqa: D102
        res = {}
        for name, field in self.fields.items():
            res[name] = f"{field.long_name}  [{field.units}]"
        return res

    @property
    def fields(self) -> OrderedDict[str, fr.ScalarField]:
        """The dictionary of scalar fields."""
        return self._fields

    @fields.setter
    def fields(self, value: OrderedDict[str, fr.ScalarField]) -> None:
        # check that all fields have the same spectral flag
        if len({field.is_spectral for field in value.values()}) != 1:
            msg = "All fields must have the same spectral flag"
            raise ValueError(msg)
        self._fields = value

    @property
    def field_list(self) -> list[fr.ScalarField]:
        """The list of scalar fields."""
        return list(self.fields.values())

    @property
    def vector_dim(self) -> int:
        """The vector dimension."""
        # should be read-only
        return self._vector_dim

    @property
    def is_spectral(self) -> bool:  # noqa: D102
        return next(iter(self.fields.values())).is_spectral

    # ================================================================
    #  Shrinking operations
    # ================================================================

    def sum(self: T, axes: tuple[int] | None = None) -> T:  # noqa: D102
        self.apply_elementwise(self, lambda field: field.sum(axes))

    def max(self: T, axes: tuple[int] | None = None) -> T:  # noqa: D102
        self.apply_elementwise(self, lambda field: field.max(axes))

    def min(self: T, axes: tuple[int] | None = None) -> T:  # noqa: D102
        self.apply_elementwise(self, lambda field: field.min(axes))

    def integrate(self: T, axes: tuple[int] | None = None) -> T:  # noqa: D102
        self.apply_elementwise(self, lambda field: field.integrate(axes))

    # ================================================================
    #  Arithmetic Operations
    # ================================================================

    def norm_l2(self) -> float:
        r"""
        Calculate the L2 norm of the vector field.

        Description
        -----------
        The L2 norm of the vector field :math:`\boldsymbol{z}` is defined as:

        .. math::
            ||\boldsymbol{z}||_2 = \sqrt{
                \int \boldsymbol{z} \cdot \boldsymbol{z} \, dV

        in practice, the integral is calculated as a sum over the grid cells
        and :math:`dV` is the cell volume of each grid cell.

        Returns
        -------
        float
            The L2 norm of the vector field

        """
        ncp = fr.config.ncp
        cell_volume = self.grid.dV
        return ncp.sqrt(ncp.sum(self.dot(self).unpad()) * cell_volume)

    def norm_of_diff(self, other: VectorField) -> float:
        r"""
        Norm of difference between two vector fields.

        Description
        -----------
        The norm of difference computes the normalized difference between
        two vector fields :math:`\boldsymbol{z}` and :math:`\boldsymbol{z}'`.
        It is defined as:

        .. math::
            2 \frac{||\boldsymbol{z} - \boldsymbol{z}'||}
                   {||\boldsymbol{z}|| + ||\boldsymbol{z}'||}

        where :math:`||\cdot||` is the L2 norm of the vector field. The norm of
        difference is in the range [0, 2].

        Parameters
        ----------
        other : VectorField
            The other vector field to compare with

        Returns
        -------
        float
            The norm of difference between the two vector fields

        """
        return 2 * (self - other).norm_l2() / (self.norm_l2() + other.norm_l2())

    def dot(self: T,  # noqa: D102
            other: fr.ScalarField | VectorField | fr.TensorField,
            ) -> fr.ScalarField | VectorField | T:
        # check that the spectral flag is the same
        if self.is_spectral != other.is_spectral:
            msg = "Cannot take dot product of spectral and real fields"
            raise ValueError(msg)
        # complex conjugate the other field if it is spectral
        if other.is_spectral:
            other = other.conj()

        if isinstance(other, fr.ScalarField):
            return self * other
        if isinstance(other, fr.VectorField):
            return sum((self * other).fields.values())
        if isinstance(other, fr.TensorField):
            msg = "Dot product of vector field with tensor field not possible"
            raise TypeError(msg)
        msg = f"Invalid type for dot product: {type(other)}"
        raise TypeError(msg)

    def conj(self: T) -> T:  # noqa: D102
        return self.apply_elementwise(self, lambda field: field.conj())

    @staticmethod
    def apply_elementwise(vector_field: T,
                          op: Callable[[fr.ScalarField], fr.ScalarField],
                          ) -> T:
        """
        Apply an operation elementwise to the vector field.

        Description
        -----------
        This method applies an operation elementwise to each scalar field
        in the vector field and returns a new vector field with the modified
        fields.

        Parameters
        ----------
        vector_field : VectorField
            The vector field to apply the operation to.
        op : Callable[[fr.ScalarField], fr.ScalarField]
            The operation to apply to each scalar field. Should take a scalar
            field as input and return a scalar field.

        Returns
        -------
        VectorField
            The new vector field with the modified fields

        """
        cls = vector_field.__class__
        # apply the operation to each field
        new_fields = OrderedDict(
            (name, op(field)) for name, field in vector_field.fields.items())
        return cls(vector_field.mset,
                   field_list=new_fields,
                   vector_dim=vector_field.vector_dim)

    @staticmethod
    def _apply_operation(op: Callable[[T, any], fr.FieldBase],
                         field: T,
                         other: any) -> T:
        cls = field.__class__
        if isinstance(other, fr.VectorField):
            names = list(field.fields)
            fields = OrderedDict(
                (name, op(field.fields[name], other.fields[name]))
                for name in names)
            return cls(field.mset, field_list=fields, vector_dim=field.vector_dim)
        if isinstance(other, (fr.ScalarField, float, int, complex, np.number)):
            return field.apply_elementwise(field, lambda x: op(x, other))
        msg = "Operation not supported for the given type"
        raise TypeError(msg)

