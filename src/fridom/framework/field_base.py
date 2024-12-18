"""Base class for all types of fields."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

T = TypeVar("T", bound="FieldBase")

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
        self._mset = mset

    # ================================================================
    #  General Methods
    # ================================================================

    @abstractmethod
    def fft(self: T,
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            ) -> T:
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

    @abstractmethod
    def ifft(self: T,
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             ) -> T:
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
            raise ValueError(msg)

    @abstractmethod
    def sync(self: T) -> T:
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
    def apply_water_mask(self: T) -> T:
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
    def set_random(self: T, seed: int = 1234) -> T:
        r"""
        Set the field to random values.

        Description
        -----------
        This method sets the field to random values. If the field is in spectral
        space, the random values are complex.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.

        Returns
        -------
        FieldBase
            The field with random values.

        """


    @abstractmethod
    def __copy__(self: T) -> T:
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
        return res

    # ================================================================
    #  Differential Operators
    # ================================================================

    @abstractmethod
    def diff(self: T,
             axis: int,
             order: int = 1,
             ) -> T:
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

    @abstractmethod
    def grad(self,
             axes: list[int] | None = None,
             ) -> fr.VectorField | fr.TensorField:
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

    @abstractmethod
    def laplacian(self: T,
                  axes: tuple[int] | None = None,
                  ) -> T:
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

    @abstractmethod
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

    # ================================================================
    #  xarray Interface
    # ================================================================

    @property
    @abstractmethod
    def xr(self) -> xr.DataArray | xr.Dataset:
        r"""
        The xarray representation of the field.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The xarray representation of the field.

        """

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

    @classmethod
    @abstractmethod
    def from_xarray(cls: type[T],
                    mset: fr.ModelSettingsBase,
                    ds: xr.DataArray | xr.Dataset,
                    ) -> T:
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
    def from_netcdf(cls: type[T],
                    mset: fr.ModelSettingsBase, path: str) -> T:
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

    @property
    def grid(self) -> fr.grid.GridBase:
        """The grid object."""
        return self.mset.grid

    @property
    @abstractmethod
    def is_spectral(self) -> bool:
        """Flag indicating whether the field is in spectral space."""

    # ================================================================
    #  Shrink / Extend operations
    # ================================================================

    @abstractmethod
    def extend(self: T, topo: tuple[bool]) -> T:
        r"""
        Extend the field in the specified directions.

        Description
        -----------
        This method extends the field in the specified directions. The field
        can be extended in any direction, but it cannot be shrunk. This means
        that if the field is extended in a direction, it has to be extended in
        all directions. Values in the extended directions are copied from the
        original field, such that:

        .. math::
            f_{\text{new}}(x, y, z) = f_{\text{old}}(x, y)

        where :math:`f_{\text{new}}` is the new field extended in (x, y, z),
        and :math:`f_{\text{old}}` is the old field, extended in (x, y).

        Parameters
        ----------
        topo : tuple[bool]
            The new topology of the field.

        Returns
        -------
        FieldBase
            The extended field.

        Raises
        ------
        ValueError
            If the field is shrunk in any direction.

        """

    @abstractmethod
    def sum(self: T, axes: tuple[int] | None = None) -> T:
        """
        Sum of the Field over the whole domain in the specified axes.

        Description
        -----------
        This method computes the sum of the Field over the whole domain
        (across all processes) in the specified axes. If no axes are specified,
        the sum is computed over all axes.

        .. note::
            We recommend using the `f.integrate()` method to integrate the field
            in certain directions. The `integrate()` method takes the grid spacing
            into account while the `sum()` method does not.

        Parameters
        ----------
        axes : tuple[int] | None
            The axes to sum over. If None, sum over all axes.

        Returns
        -------
        FieldBase
            The sum of the field. The returned field has no extend in the
            specified axes.

        """

    @abstractmethod
    def max(self: T, axes: tuple[int] | None = None) -> T:
        """
        Maximum value of the Field over the whole domain.

        Description
        -----------
        This method computes the maximum value of the Field over the whole
        domain (across all processes) in the specified axes. If no axes are
        specified, the maximum is computed over all axes.

        Parameters
        ----------
        axes : tuple[int] | None
            The axes to compute the maximum over. If None, compute the maximum
            over all axes.

        Returns
        -------
        FieldBase
            The maximum value of the Field over the specified axes. The
            returned field has no extend in the specified axes.

        """

    @abstractmethod
    def min(self: T, axes: tuple[int] | None = None) -> T:
        """
        Minimum value of the Field over the whole domain.

        Description
        -----------
        This method computes the minimum value of the Field over the whole
        domain (across all processes) in the specified axes. If no axes are
        specified, the minimum is computed over all axes.

        Parameters
        ----------
        axes : tuple[int] | None
            The axes to compute the minimum over. If None, compute the minimum
            over all axes.

        Returns
        -------
        FieldBase
            The minimum value of the Field over the specified axes. The
            returned field has no extend in the specified axes.

        """

    @abstractmethod
    def integrate(self: T, axes: tuple[int] | None = None) -> T:
        r"""
        Global integral of the Field in specified axes.

        Description
        -----------
        Computes the global integral of the Field in the specified axes:

        .. math::
            \sum_{i} \int_{x_i} f(\boldsymbol{x}) dx_i

        If no axes are specified, the integral is computed over all axes and a
        scalar (float) is returned. If axes are specified, the integral is computed
        over the specified axes and a new Field that is shrinked in the specified
        axes is returned.

        Parameters
        ----------
        axes : tuple[int] | None
            The axes to integrate over. If None, integrate over all axes.

        Returns
        -------
        FieldBase
            The integral of the Field over the specified axes.

        """

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
    def conj(self: T) -> T:
        r"""
        Compute the complex conjugate.

        Returns
        -------
        FieldBase
            The complex conjugate. If the field is real, the field itself is returned.

        """

    @staticmethod
    @abstractmethod
    def _apply_operation(
        op: Callable[[T, any], T],
        field: T,
        other: any) -> T: ...

    def __add__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: x + y, self, other)

    def __radd__(self: T, other: any) -> T:
        return self.__add__(other)

    def __sub__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: x - y, self, other)

    def __rsub__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: y - x, self, other)

    def __mul__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: x * y, self, other)

    def __rmul__(self: T, other: any) -> T:
        return self.__mul__(other)

    def __truediv__(self: T, other: any) -> T:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: x / y, self, other)

    def __rtruediv__(self: T, other: any) -> T:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._apply_operation(lambda x, y: y / x, self, other)

    def __pow__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: x ** y, self, other)

    def __rpow__(self: T, other: any) -> T:
        return self._apply_operation(lambda x, y: y ** x, self, other)

    def __matmul__(self, other: FieldBase) -> FieldBase:
        return self.dot(other)

    def __neg__(self: T) -> T:
        return self._apply_operation(lambda x, _: -x, self, None)
