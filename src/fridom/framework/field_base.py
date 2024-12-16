"""Base class for all types of fields."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:
    import xarray as xr

class FieldBase:

    r"""
    Base class for all types of fields.

    Description
    -----------
    A field is a mathematical mapping from the grid space :math:`\Omega`
    to an abstract space :math:`\mathcal{F}`. This abstract space can for
    example be the real or complex numbers for scalar fields, or the
    space of vectors or tensors for vector or tensor fields.

    This base class defines the interface for all types of fields.

    Parameters
    ----------
    mset : fr.ModelSettingsBase
        The model settings.

    """

    def __init__(self, mset: fr.ModelSettingsBase) -> None:
        self.mset = mset

    # ================================================================
    #  General Methods
    # ================================================================

    @abstractmethod
    def fft(self,
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            ) -> FieldBase:
        r"""
        Perform a Fast Fourier Transform (FFT) on the field.

        Description
        -----------
        Computes the Fast Fourier Transform (FFT) of the field. The
        padding parameter can be used to specify the zero-padding
        strategy.

        Parameters
        ----------
        padding : fr.grid.FFTPadding
            The padding strategy.

        Returns
        -------
        FieldBase
            The FFT of the field.

        """
        self._fft_possible()

    @abstractmethod
    def ifft(self,
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             ) -> FieldBase:
        r"""
        Perform an Inverse Fast Fourier Transform (IFFT) on the field.

        Description
        -----------
        Computes the Inverse Fast Fourier Transform (IFFT) of the field.
        The padding parameter can be used to specify the zero-padding
        strategy.

        Parameters
        ----------
        padding : fr.grid.FFTPadding
            The padding strategy.

        Returns
        -------
        FieldBase
            The IFFT of the field.

        """
        self._ifft_possible()

    def _fft_possible(self) -> None:
        r"""
        Check if a Fast Fourier Transform (FFT) is possible.

        Description
        -----------
        This method checks if a Fast Fourier Transform (FFT) is possible
        for the field. This is the case if the field is not already in
        spectral space and the grid allows for FFTs.

        Raises
        ------
        ValueError
            If the field is already in spectral space.
        NotImplementedError
            If the grid does not allow for FFTs.

        """
        if not self.grid.fourier_transform_available:
            msg = "Fourier transform not available for this grid"
            raise NotImplementedError(msg)

        if self.is_spectral:
            msg = "Field is in spectral space, cannot perform fft"
            raise ValueError(msg)

    def _ifft_possible(self) -> None:
        r"""
        Check if an Inverse Fast Fourier Transform (IFFT) is possible.

        Description
        -----------
        This method checks if an Inverse Fast Fourier Transform (IFFT) is
        possible for the field. This is the case if the field is in
        spectral space and the grid allows for FFTs.

        Raises
        ------
        ValueError
            If the field is not in spectral space.
        NotImplementedError
            If the grid does not allow for FFTs.

        """
        if not self.grid.fourier_transform_available:
            msg = "Fourier transform not available for this grid"
            raise NotImplementedError(msg)

        if not self.is_spectral:
            msg = "Field is not in spectral space, cannot perform ifft"
            raise ValueError

    @abstractmethod
    def sync(self) -> FieldBase:
        r"""
        Synchronize the field across all MPI ranks and apply boundary conditions.

        Description
        -----------
        This method synchronizes the field across all MPI ranks and applies
        the boundary conditions. This is necessary to ensure that the ghost
        cells are up-to-date. This method changes the field in-place, but
        also returns the synchronized field.

        Returns
        -------
        FieldBase
            The synchronized field.

        """

    @abstractmethod
    def apply_water_mask(self) -> FieldBase:
        """
        Apply a water mask to the field.

        Description
        -----------
        A water mask is a binary field that indicates which cells are water
        (active) and which are land (inactive). This method applies the water
        mask to the field. The field is changed in-place.

        Returns
        -------
        FieldBase
            The field with the water mask applied.

        """

    @abstractmethod
    def has_nan(self) -> bool:
        r"""
        Check if the field contains NaN values.

        Returns
        -------
        bool
            Flag indicating whether the field contains NaN values.

        """

    @abstractmethod
    def __copy__(self) -> FieldBase:
        r"""
        Create a copy of the field.

        Description
        -----------
        Child classes should implement this method to ensure that the content
        of the field is copied, but not the model settings.

        Returns
        -------
        FieldBase
            A copy of the field.

        """

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}("
        for key, value in self.info.items():
            res += f"\n  {key}={value}, "
        res += "\n)"

    # ================================================================
    #  Differential Operators
    # ================================================================

    def diff(self,
             axis: int,
             order: int = 1,
             ) -> fr.ScalarField | fr.VectorField | fr.TensorField:
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
        fr.ScalarField | fr.VectorField | fr.TensorField
            The derivative of the field along the specified axis.

        """
        return self.grid.diff_module.diff(self, axis, order)

    def grad(self,
             axes: list[int] | None = None,
             ) -> FieldBase:
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
        fr.VectorField | fr.TensorField
            The gradient of the field along the specified axes. The list contains
            the gradient components along each axis. Axis which are not included
            in `axes` will have a value of `None`.
            E.g. for a 3D grid, `diff.grad(f, axes=[0, 2])` will return
            `[df/dx, None, df/dz]`.

        """
        return self.grid.diff_module.grad(self, axes)

    def laplacian(self,
                  axes: tuple[int] | None = None,
                  ) -> fr.ScalarField | fr.VectorField | fr.TensorField:
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
        fr.ScalarField | fr.VectorField | fr.TensorField
            The Laplacian of the field.

        """
        return self.grid.diff_module.laplacian(self, axes)

    def div(self) -> fr.ScalarField | fr.VectorField:
        r"""
        Compute the divergence.

        .. math::
            \nabla \cdot f = \sum_{i=1}^n \partial_i f

        Returns
        -------
        fr.ScalarField | fr.VectorField
            The divergence of the field.

        """
        return self.grid.diff_module.div(vec=self)


    # ================================================================
    #  xarray Interface
    # ================================================================
    @property
    def xr(self) -> xr.DataArray | xr.Dataset:
        r"""
        The xarray representation of the field.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The xarray representation of the field.

        """
        return self.xrs[:]

    @property
    @abstractmethod
    def xrs(self) -> fr.utils.SliceableAttribute[xr.DataArray | xr.Dataset]:
        """
        Convert a slice of the field to an xarray object.

        Description
        -----------
        This method returns a sliceable attribute that allows to convert
        a slice of the field to an xarray object. This is useful when dealing
        with large fields and only a subset of the data is needed. For example,
        the top region of the field.

        """

    @abstractmethod
    @classmethod
    def from_xarray(cls,
                    mset: fr.ModelSettingsBase,
                    ds: xr.DataArray | xr.Dataset,
                    ) -> FieldBase:
        """
        Create a field from an xarray object.

        Description
        -----------
        This method creates a field from an xarray object. The model settings
        are required to create the field.

        Parameters
        ----------
        mset : fr.ModelSettingsBase
            The model settings.
        ds : xr.DataArray | xr.Dataset
            The xarray object.

        Returns
        -------
        FieldBase
            The field.

        """

    def to_netcdf(self, path: str) -> None:
        r"""
        Save the field to a NetCDF file.

        Description
        -----------
        This method saves the field to a NetCDF file.

        Parameters
        ----------
        path : str
            The path to the NetCDF file.

        """
        self.xr.to_netcdf(path, auto_complex=True)

    @classmethod
    def from_netcdf(cls, mset: fr.ModelSettingsBase, path: str) -> FieldBase:
        r"""
        Create a field from a NetCDF file.

        Parameters
        ----------
        mset : fr.ModelSettingsBase
            The model settings.
        path : str
            The path to the NetCDF file.

        Returns
        -------
        FieldBase
            The field.

        """
        import xarray as xr
        ds = xr.open_dataset(path)
        return cls.from_xarray(mset, ds)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    @abstractmethod
    def info(self) -> dict:
        """Dictionary with information about the field."""

    @property
    def mset(self) -> fr.ModelSettingsBase:
        """The model settings."""
        return self._mset

    @mset.setter
    def mset(self, value: fr.ModelSettingsBase) -> None:
        self._mset = value

    @property
    def grid(self) -> fr.grid.GridBase:
        """The grid object."""
        return self.mset.grid

    @property
    @abstractmethod
    def is_spectral(self) -> bool:
        """Flag indicating whether the field is in spectral space."""

    # ================================================================
    #  Arithmetic Operations
    # ================================================================
    @abstractmethod
    def dot(self, other: FieldBase) -> FieldBase:
        r"""
        Compute the dot product with another field.

        Parameters
        ----------
        other : FieldBase
            The other field.

        Returns
        -------
        FieldBase
            The dot product.

        Description
        -----------
        Computes the dot product with another field. The dot product is
        defined as

        .. math::
            f \cdot g^*

        where :math:`f` and :math:`g` are the fields and :math:`^*` denotes
        the complex conjugate.

        The return value depends on the type of the fields. The following
        table shows the possible return values:

        +-------------------+-------------------+-------------------+
        | Field Type        | Field Type        | Return Type       |
        +===================+===================+===================+
        | ScalarField       | ScalarField       | ScalarField       |
        +-------------------+-------------------+-------------------+
        | ScalarField       | VectorField       | VectorField       |
        +-------------------+-------------------+-------------------+
        | ScalarField       | TensorField       | TensorField       |
        +-------------------+-------------------+-------------------+
        | VectorField       | ScalarField       | VectorField       |
        +-------------------+-------------------+-------------------+
        | VectorField       | VectorField       | ScalarField       |
        +-------------------+-------------------+-------------------+
        | VectorField       | TensorField       | Error             |
        +-------------------+-------------------+-------------------+
        | TensorField       | ScalarField       | TensorField       |
        +-------------------+-------------------+-------------------+
        | TensorField       | VectorField       | VectorField       |
        +-------------------+-------------------+-------------------+
        | TensorField       | TensorField       | TensorField       |
        +-------------------+-------------------+-------------------+

        """

    @abstractmethod
    def conj(self) -> FieldBase:
        r"""
        Compute the complex conjugate.

        Returns
        -------
        FieldBase
            The complex conjugate. If the field is real, the field itself is returned.

        """

    @abstractmethod
    @staticmethod
    def _apply_operation(
        op: Callable[[FieldBase, any], FieldBase],
        field: FieldBase,
        other: any) -> FieldBase: ...

    def __add__(self, other: any) -> FieldBase:
        return self._apply_operation(lambda x, y: x + y, self, other)

    def __radd__(self, other: any) -> FieldBase:
        return self.__add__(other)

    def __sub__(self, other: any) -> FieldBase:
        return self._apply_operation(lambda x, y: x - y, self, other)

    def __rsub__(self, other: any) -> FieldBase:
        return self._apply_operation(lambda x, y: y - x, self, other)

    def __mul__(self, other: any) -> FieldBase:
        return self._apply_operation(lambda x, y: x * y, self, other)

    def __rmul__(self, other: any) -> FieldBase:
        return self.__mul__(other)

    def __truediv__(self, other: any) -> FieldBase:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: x / y, self, other)

    def __rtruediv__(self, other: any) -> FieldBase:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: y / x, self, other)

    def __pow__(self, other: any) -> FieldBase:
        return self._apply_operation(lambda x, y: x ** y, self, other)

    def __matmul__(self, other: FieldBase) -> FieldBase:
        return self.dot(other)

    def __neg__(self) -> FieldBase:
        return self._apply_operation(lambda x, _: -x, self, None)
