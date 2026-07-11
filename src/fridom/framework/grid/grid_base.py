"""Base class for all grids in the framework."""
from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Literal

import fridom.framework as fr

if TYPE_CHECKING:
    from numpy import ndarray


@partial(fr.utils.jaxify,
         dynamic=("_x_mesh", "_x_global", "_k_mesh", "_k_global"))
class GridBase:

    """
    Base class for all grids in the framework.

    Description
    -----------
    This class does not implement any functionality, but provides a template
    for all grid classes in the framework. The base class also sets default
    flags and attributes that are common to all grids. Child classes should
    override these flags and attributes as needed.

    Flags
    -----
    fourier_transform_available : bool
        Indicates whether the grid supports fast fourier transforms.
    mpi_available : bool
        Indicates whether the grid supports MPI parallelization.
    """

    def __init__(self, n_dims: int) -> None:

        self.name = "GridBase"

        self._n_dims = n_dims
        self._shape = None
        self._domain_size = None
        self._total_grid_points = None
        self._periodic_bounds = None
        self._x_mesh = None
        self._x_global = None
        self._x_local = None
        self._dx = None
        self._cell_volume = None
        self._mset = None
        self._water_mask = fr.grid.WaterMask()
        # The domain decomposition
        self._domain_decomposition = None
        # The cell center
        center = fr.grid.AxisPosition.CENTER
        self._cell_center = fr.grid.Position(tuple([center] * n_dims))
        # spectral properties
        self._k_mesh = None
        self._k_global = None
        self._k_local = None
        self._omega_analytical = None
        self._omega_space_discrete = None
        self._omega_time_discrete = None
        # operator modules
        self._diff_module: fr.grid.DiffModule = None
        self._interp_module: fr.grid.InterpolationModule = None

        # prepare for numpy conversion (the numpy copy will be stored here)
        self._cpu = None

        # ---------------------------------------------------------------------
        #  Set default flags
        # ---------------------------------------------------------------------
        self._fourier_transform_available = False
        self._mpi_available = False


    def setup(self, mset: fr.ModelSettingsBase) -> None:
        """
        Initialize the grid from the model settings.

        Parameters
        ----------
        mset : ModelSettingsBase
            The model settings object. This is for example needed to
            determine the required halo size.
        """
        self._diff_module.setup(mset=mset)
        self._interp_module.setup(mset=mset)

    def get_mesh(self,
                 position: fr.grid.Position | None = None,
                 spectral: bool = False
    ) -> tuple[ndarray]:
        """
        Get the meshgrid of the grid points.

        Parameters
        ----------
        position : Position | None, optional
            The position of the field (default: None).
        spectral : bool, optional
            Whether to return the meshgrid of the spectral domain (default:
            False).

        Returns
        -------
        `tuple[ndarray]`
            The meshgrid of the grid points.
        """
        if position is None:
            position = self.cell_center
        if position != self.cell_center:
            raise NotImplementedError("Not implemented for this grid")
        if spectral:
            return self._k_mesh
        return self._x_mesh

    # ----------------------------------------------------------------
    #  Fourier Transform Methods
    # ----------------------------------------------------------------

    @abstractmethod
    def fft(self,
             arr: ndarray,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             axes: tuple[int] | None = None,
            ) -> ndarray:
        """
        Perform a (fast) fourier transform on the input array.

        Parameters
        ----------
        arr : ndarray
            The input array.
        bc_types : tuple[BCType] | None, optional
            The boundary conditions to apply to each axis (default: None).
        positions : tuple[AxisPosition] | None, optional
            The position of the field (default: None).
        axes : tuple[int] | None, optional
            The axes to transform (default: None).

        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    @abstractmethod
    def ifft(self,
             arr: ndarray,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             axes: tuple[int] | None = None,
             ) -> ndarray:
        """
        Perform an inverse (fast) fourier transform on the input array.

        Parameters
        ----------
        arr : ndarray
            The input array.
        bc_types : tuple[BCType] | None, optional
            The boundary conditions to apply to each axis (default: None).
        positions : tuple[AxisPosition] | None, optional
            The position of the field (default: None).
        axes : tuple[int] | None, optional
            The axes to transform (default: None).

        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    #  Spectral Analysis Tools
    # ----------------------------------------------------------------

    @abstractmethod
    def omega(self,
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        """
        Compute the dispersion relation of the model.

        Parameters
        ----------
        k : tuple[float] | tuple[ndarray]
            The wave numbers
        use_discrete : bool, optional
            Whether to include space-discretization effects (default: False).

        Returns
        -------
        `ndarray`
            The dispersion relation (omega(k)).
        """
        raise NotImplementedError

    @abstractmethod
    def vec_q(self, s: int, use_discrete: bool = True) -> fr.VectorField:
        """
        Compute the eigenvector of the linear operator of the mode `s`.

        Parameters
        ----------
        s : int
            The mode (which eigenvalue / eigenvector to compute).
        use_discrete : bool, optional
            Whether to include space-discretization effects (default: True).

        Returns
        -------
        `fr.VectorField`
            The eigenvector of the linear operator.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_p(self, s: int, use_discrete: bool = True) -> fr.VectorField:
        """
        Compute the projection vector of the linear operator of the mode `s`.

        Parameters
        ----------
        s : int
            The mode (which eigenvalue / eigenvector to compute).
        use_discrete : bool, optional
            Whether to include space-discretization effects (default: True).

        Returns
        -------
        `fr.VectorField`
            The projection vector of the linear operator.

        """
        raise NotImplementedError

    @property
    def omega_analytical(self) -> ndarray:
        """Analytical dispersion relation."""
        if self._omega_analytical is None:
            self._omega_analytical = self.omega(
                self.k_mesh, use_discrete=False)
        return self._omega_analytical

    @property
    def omega_space_discrete(self) -> ndarray:
        """Dispersion relation with space-discretization effects."""
        if self._omega_space_discrete is None:
            self._omega_space_discrete = self.omega(
                self.k_mesh, use_discrete=True)

        return self._omega_space_discrete

    @property
    def omega_time_discrete(self) -> ndarray:
        """
        Dispersion relation with space-time-discretization effects.

        Warning: The computation may be very slow.
        """
        if self._omega_time_discrete is None:
            om_space_discrete = self.omega_space_discrete
            ts = self.mset.time_stepper
            om = ts.time_discretization_effect(om_space_discrete)
            self._omega_time_discrete = om
        return self._omega_time_discrete

    # ----------------------------------------------------------------
    #  Domain Decomposition Methods
    # ----------------------------------------------------------------

    def sync(self,
             arr: ndarray,
             flat_axes: list[int] | None = None) -> ndarray:
        """
        Synchronize the halo (boundary) points of an array across ranks.

        Parameters
        ----------
        arr : ndarray
            The array to synchronize.

        Returns
        -------
        `ndarray`
            The synchronized array.
        """
        return self.domain_decomp.sync(arr, flat_axes=flat_axes)

    def sync_multi(self, arrs: tuple[ndarray]) -> tuple[ndarray]:
        """
        Synchronize the halo points of multiple arrays across ranks.

        Parameters
        ----------
        arrs : list[ndarray]
            The list of arrays to synchronize.

        Returns
        -------
        `list[ndarray]`
            The synchronized list of arrays.
        """
        return self.domain_decomp.sync_multiple(arrs)

    def unpad(self, arr: ndarray) -> ndarray:
        """
        Remove the halo padding from an array.

        Parameters
        ----------
        arr : ndarray
            The padded array.

        Returns
        -------
        `ndarray`
            The unpadded array.
        """
        return self.domain_decomp.unpad(arr)

    def pad(self, arr: ndarray) -> ndarray:
        """
        Add halo padding to an array.

        Parameters
        ----------
        arr : ndarray
            The unpadded array.

        Returns
        -------
        `ndarray`
            The padded array.
        """
        return self.domain_decomp.pad(arr)

    def create_array(self,
                     pad: bool = True,
                     spectral: bool = False,
                     topo: tuple[bool] | None = None) -> ndarray:
        """
        Create an array.

        Parameters
        ----------
        pad : bool
            Whether to add padding to the array.
        spectral : bool
            Whether the array is in spectral space.
        topo : tuple[bool] | None
            The topology of the array. Axes with false are flat
            (only one grid point)
        """
        return self.domain_decomp.create_array(
            pad=pad, spectral=spectral, topo=topo)

    def create_random_array(self,
                            seed: int = 1234,
                            pad: bool = True,
                            spectral: bool = False,
                            topo: tuple[bool] | None = None
                            ) -> ndarray:
        """
        Create a random array.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        pad : bool
            Whether to add padding to the array.
        spectral : bool
            Whether the array is in spectral space.
        topo : tuple[bool] | None
            The topology of the array. Axes with false are flat
            (only one grid point)
        """
        return self.domain_decomp.create_random_array(
            seed=seed, pad=pad, spectral=spectral, topo=topo)

    # ----------------------------------------------------------------
    #  Shrink / Extend Methods
    # ----------------------------------------------------------------

    @abstractmethod
    def sum(self,
            field: fr.ScalarField,
            axes: tuple[int] | None = None) -> fr.ScalarField:
        """
        Sum a field over the given axes.

        Parameters
        ----------
        field : ScalarField
            The field to sum.
        axes : tuple[int] | None, optional
            The axes to sum over. If None, all axes are summed over (default:
            None).

        Returns
        -------
        ScalarField
            The summed field with no extend in the summed axes.

        """
        raise NotImplementedError

    @abstractmethod
    def min(self,
            field: fr.ScalarField,
            axes: tuple[int] | None = None) -> fr.ScalarField:
        """
        Compute the minimum of a field over the given axes.

        Parameters
        ----------
        field : ScalarField
            The field to compute the minimum.
        axes : tuple[int] | None, optional
            The axes to compute the minimum over. If None, all axes are used
            (default: None).

        Returns
        -------
        ScalarField
            The field with the minimum value in the given axes.

        """
        raise NotImplementedError

    @abstractmethod
    def max(self,
            field: fr.ScalarField,
            axes: tuple[int] | None = None) -> fr.ScalarField:
        """
        Compute the maximum of a field over the given axes.

        Parameters
        ----------
        field : ScalarField
            The field to compute the maximum.
        axes : tuple[int] | None, optional
            The axes to compute the maximum over. If None, all axes are used
            (default: None).

        Returns
        -------
        ScalarField
            The field with the maximum value in the given axes.

        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self,
                  field: fr.ScalarField,
                  axes: tuple[int] | None = None) -> fr.ScalarField:
        """
        Integrate a scalar field over a given domain.

        Parameters
        ----------
        field : ScalarField
            The field to integrate.
        axes : tuple[int] | None, optional
            The axes to integrate over (default: None).

        Returns
        -------
        ScalarField
            The integrated field with no extend in the integrated axes.

        """
        raise NotImplementedError

    @abstractmethod
    def cumulative_integral(self,
                            field: fr.ScalarField,
                            axis: int,
                            direction: Literal[
                                "forward", "backward"] = "forward",
                            ) -> fr.ScalarField:
        r"""
        Compute the cumulative integral of a field along a given axis.

        Description
        -----------
        The cumulative integral computes the integral starting at one end of
        the domain and accumulates the integral along the specified axis. The
        integral is computed in either the forward or backward direction.

        Forward integral:

        .. math::
            F(x) = \int_{x_0}^{x} f(x') dx'

        with axis :math:`x` and :math:`x_0` the lower bound of the domain.

        Backward integral:

        .. math::
            F(x) = \int_{x}^{x_1} f(x') dx'

        with axis :math:`x` and :math:`x_1` the upper bound of the domain.


        Parameters
        ----------
        field : ScalarField
            The field to integrate.
        axis : int
            The axis to integrate over.
        direction : str (default is "forward")
            The direction of the integration. Can be "forward" or "backward".


        Returns
        -------
        ScalarField
            The cumulative integral of the field along the given axis.

        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    #  Display methods
    # ----------------------------------------------------------------

    @property
    def info(self) -> dict:
        """
        Return a dictionary with information about the grid.

        Description
        -----------
        This method should be overridden by the child class to return a
        dictionary with information about the grid. This information is
        used to print the grid in the `__repr__` method.
        """
        return {}

    def __repr__(self) -> str:
        """Return a string representation of the grid."""
        res = self.name
        for key, value in self.info.items():
            res += f"\n  - {key}: {value}"
        return res

    # ----------------------------------------------------------------
    #  Grid Modules
    # ----------------------------------------------------------------

    @property
    def diff_module(self) -> fr.grid.DiffModule:
        """The differential operator module."""
        return self._diff_module

    @diff_module.setter
    def diff_module(self, value: fr.grid.DiffModule) -> None:
        if not isinstance(value, fr.grid.DiffModule):
            raise TypeError(
                "The differential operator module must be a DiffBase "
                "object")
        self._diff_module = value

    @property
    def interp_module(self) -> fr.grid.InterpolationModule:
        """The interpolation operator module."""
        return self._interp_module

    @interp_module.setter
    def interp_module(self, value: fr.grid.InterpolationModule) -> None:
        if not isinstance(value, fr.grid.InterpolationModule):
            raise TypeError(
                "The interpolation operator module must be an "
                "InterpolationBase object")
        self._interp_module = value

    @property
    def water_mask(self) -> fr.grid.WaterMask:
        """Get the water mask."""
        return self._water_mask

    @water_mask.setter
    def water_mask(self, value: fr.grid.WaterMask) -> None:
        self._water_mask = value

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def mset(self) -> fr.ModelSettingsBase | None:
        """The model settings object."""
        return self._mset

    @property
    def domain_decomp(self) -> fr.domain_decomposition.DomainDecomposition:
        """The domain decomposition object."""
        return self._domain_decomp

    @property
    def halo(self) -> int:
        """
        The halo size of the grid.

        To change the halo size of the grid, modify the halo attribute of the
        model settings.
        """
        return self.domain_decomp.halo

    @property
    def n_dims(self) -> int:
        """The number of dimensions of the grid."""
        return self._n_dims

    @property
    def shape(self) -> tuple[int]:
        """The number of grid points in each dimension."""
        return self._shape

    @property
    def domain_size(self) -> tuple[float]:
        """The length of the grid in each dimension."""
        return self._domain_size

    @property
    def total_grid_points(self) -> int:
        """The total number of grid points in the grid."""
        return self._total_grid_points

    @property
    def periodic_bounds(self) -> list[bool]:
        """Tuple of booleans indicating periodicity in each dimension."""
        return self._periodic_bounds

    @property
    def cell_center(self) -> fr.grid.Position:
        """The position of the cell centers."""
        return self._cell_center

    @property
    def x_mesh(self) -> tuple[ndarray]:
        """The meshgrid of the grid points."""
        return self._x_mesh

    @property
    def x_global(self) -> tuple[ndarray]:
        """The x-vector of the global grid points."""
        return self._x_global

    @property
    def k_mesh(self) -> ndarray:
        """The wavenumber of the grid."""
        return self._k_mesh

    @property
    def k_global(self) -> ndarray:
        """The global wavenumber of the grid."""
        return self._k_global

    @property
    def dx(self) -> tuple[ndarray]:
        """The grid spacing in each dimension."""
        return self._dx

    @property
    def cell_volume(self) -> ndarray:
        """The volume element of the grid."""
        return self._cell_volume

    @property
    def characteristic_function(self) -> fr.ScalarField:
        """
        The characteristic function of the grid (1 inside, 0 outside).

        Description
        -----------
        The characteristic function is a scalar field that is 1 inside the
        domain and 0 outside. It is useful for masking fields or for
        integrating over the domain. For example, the total volume of the
        domain can be computed as the integral of the characteristic function.

        """
        mdata = fr.FieldMetadata(name="characteristic_function",
                                 long_name="Characteristic Function",
                                 units="1",
                                 position=self.cell_center)
        f = fr.ScalarField(mset = self.mset, mdata = mdata)
        f += 1
        return f.apply_water_mask()

    # ================================================================
    #  Flags
    # ================================================================

    @property
    def fourier_transform_available(self) -> bool:
        """Indicates whether the grid supports fast fourier transforms."""
        return self._fourier_transform_available

    @fourier_transform_available.setter
    def fourier_transform_available(self, value: bool) -> None:
        self._fourier_transform_available = value

    @property
    def mpi_available(self) -> bool:
        """Indicates whether the grid supports MPI parallelization."""
        return self._mpi_available

    @mpi_available.setter
    def mpi_available(self, value: bool) -> None:
        self._mpi_available = value
