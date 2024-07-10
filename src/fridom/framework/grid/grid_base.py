# Import external modules
from typing import TYPE_CHECKING
import numpy as np
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.field_variable import FieldVariable

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
    `fourier_transform_available` : `bool`
        Indicates whether the grid supports fast fourier transforms.
    `mpi_available` : `bool`
        Indicates whether the grid supports MPI parallelization.

    Attributes
    ----------
    `n_dims` : `int`
        The number of dimensions of the grid.
    `N` : `tuple[int]`
        The number of grid points in each dimension.
    `L` : `tuple[float]`
        The length of the grid in each dimension.
    `total_grid_points` : `int`
        The total number of grid points in the grid.
    `periodic_bounds` : `tuple`
        A tuple of booleans indicating whether the grid is periodic in each
        dimension.
    `inner_slice` : `tuple[slice]`
        The slice of the grid that excludes the boundary points.
    `X` : tuple(`np.ndarray`)
        The meshgrid of the grid points.
    `x_global` : tuple(`np.ndarray`)
        The x-vector of the global grid points.
    `x_local` : `np.ndarray`
        The x-vector of the local grid points.
    `dx` : tuple(`np.ndarray`)
        The grid spacing in each dimension.
    `dV` : `np.ndarray`
        The volume element of the grid.

    Methods
    -------
    `setup(mset: ModelSettingsBase)`
        Setup the grid. (must be implemented by child classes)
    `fft(f: np.ndarray) -> np.ndarray`
        Perform a (fast) fourier transform on the input array.
    `ifft(f: np.ndarray) -> np.ndarray`
        Perform an inverse (fast) fourier transform on the input array.
    `sync(f: FieldVariable)`
        Synchronize the halo (boundary) points of a field variable across all
        MPI ranks.
    """
    def __init__(self, n_dims: int) -> None:
        self._n_dims = n_dims
        self._N = None
        self._L = None
        self._total_grid_points = None
        self._periodic_bounds = None
        self._inner_slice = slice(None)
        self._X = None
        self._x_global = None
        self._x_local = None
        self._dx = None
        self._dV = None

        # prepare for numpy conversion (the numpy copy will be stored here)
        self._cpu = None

        # ---------------------------------------------------------------------
        #  Set default flags
        # ---------------------------------------------------------------------
        self._fourier_transform_available = False
        self._mpi_available = False

        return

    def setup(self, mset: 'ModelSettingsBase') -> None:
        """
        Initialize the grid from the model settings.
        
        Parameters
        ----------
        `mset` : `ModelSettingsBase`
            The model settings object. This is for example needed to
            determine the required halo size.
        """       
        raise NotImplementedError

    def fft(self, f:'np.ndarray') -> 'np.ndarray':
        """
        Perform a (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `f` : `np.ndarray`
            The input array.
        
        Returns
        -------
        `np.ndarray`
            The transformed array.
        """
        raise NotImplementedError

    def ifft(self, f:'np.ndarray') -> 'np.ndarray':
        """
        Perform an inverse (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `f` : `np.ndarray`
            The input array.
        
        Returns
        -------
        `np.ndarray`
            The transformed array.
        """
        raise NotImplementedError

    def sync(self, f: 'FieldVariable') -> None:
        """
        Synchronize the halo (boundary) points of a field variable
        across all MPI ranks.
        
        Parameters
        ----------
        `f` : `FieldVariable`
            The field variable to synchronize.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    #  Grid properties
    # ----------------------------------------------------------------

    @property
    def n_dims(self) -> int:
        """The number of dimensions of the grid."""
        return self._n_dims

    @property
    def N(self) -> tuple[int]:
        """The number of grid points in each dimension."""
        return self._N

    @property
    def L(self) -> tuple[float]:
        """The length of the grid in each dimension."""
        return self._L

    @property
    def total_grid_points(self) -> int:
        """The total number of grid points in the grid."""
        return self._total_grid_points

    @property
    def periodic_bounds(self) -> list[bool]:
        """A tuple of booleans indicating whether the grid is periodic 
        in each dimension."""
        return self._periodic_bounds

    @property
    def inner_slice(self) -> tuple[slice]:
        """The slice of the grid that excludes the boundary points."""
        return self._inner_slice

    @property
    def X(self) -> tuple[np.ndarray]:
        """The meshgrid of the grid points."""
        return self._X

    @property
    def x_global(self) -> tuple[np.ndarray]:
        """The x-vector of the global grid points."""
        return self._x_global

    @property
    def x_local(self) -> np.ndarray:
        """The x-vector of the local grid points."""
        return self._x_local

    @property
    def dx(self) -> tuple[np.ndarray]:
        """The grid spacing in each dimension."""
        return self._dx

    @property
    def dV(self) -> np.ndarray:
        """The volume element of the grid."""
        return self._dV

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