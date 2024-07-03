import numpy as np
from fridom.framework.field_variable import FieldVariable
from fridom.framework.model_settings_base import ModelSettingsBase

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
    def __init__(self, n_dims: int) -> None: ...

    # ----------------------------------------------------------------
    #  Generic methods
    # ----------------------------------------------------------------

    def setup(self, mset: ModelSettingsBase) -> None: 
        """
        Initialize the grid from the model settings.
        
        Parameters
        ----------
        `mset` : `ModelSettingsBase`
            The model settings object. This is for example needed to
            determine the required halo size.
        """

    def fft(self, f:np.ndarray) -> np.ndarray:
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

    def ifft(self, f:np.ndarray) -> np.ndarray:
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

    def sync(self, f: FieldVariable) -> None:
        """
        Synchronize the halo (boundary) points of a field variable
        across all MPI ranks.
        
        Parameters
        ----------
        `f` : `FieldVariable`
            The field variable to synchronize.
        """

    # ----------------------------------------------------------------
    #  Grid properties
    # ----------------------------------------------------------------

    @property
    def n_dims(self) -> int:
        """The number of dimensions of the grid."""

    @property
    def periodic_bounds(self) -> list[bool]:
        """A tuple of booleans indicating whether the grid is periodic 
        in each dimension."""

    @property
    def inner_slice(self) -> tuple[slice]:
        """The slice of the grid that excludes the boundary points."""

    @property
    def X(self) -> tuple[np.ndarray]:
        """The meshgrid of the grid points."""

    @property
    def x_global(self) -> tuple[np.ndarray]:
        """The x-vector of the global grid points."""

    @property
    def x_local(self) -> np.ndarray:
        """The x-vector of the local grid points."""

    @property
    def dx(self) -> tuple[np.ndarray]:
        """The grid spacing in each dimension."""

    @property
    def dV(self) -> np.ndarray:
        """The volume element of the grid."""