# Import external modules
from typing import TYPE_CHECKING
# Import type information
if TYPE_CHECKING:
    import numpy as np

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
    `X` : tuple(`np.ndarray`)
        The meshgrid of the grid points.

    Methods
    -------
    `setup(mset: ModelSettingsBase)`
        Setup the grid. (must be implemented by child classes)
    `fft(f: np.ndarray) -> np.ndarray`
        Perform a (fast) fourier transform on the input array.
    `ifft(f: np.ndarray) -> np.ndarray`
        Perform an inverse (fast) fourier transform on the input array.
    `sync_physical(f: np.ndarray)`
        Synchronize the physical space array across all MPI ranks.
    `sync_spectral(f: np.ndarray)`
        Synchronize the spectral space array across all MPI ranks.
    """
    def __init__(self) -> None:
        # prepare for numpy conversion (the numpy copy will be stored here)
        self._cpu = None

        # ---------------------------------------------------------------------
        #  Set default flags
        # ---------------------------------------------------------------------
        self.fourier_transform_available = False
        self.mpi_available = False

        return

    def setup(self) -> None:
        raise NotImplementedError

    def fft(self, f:'np.ndarray') -> 'np.ndarray':
        raise NotImplementedError

    def ifft(self, f:'np.ndarray') -> 'np.ndarray':
        raise NotImplementedError

    def sync_physical(self, f:'np.ndarray') -> None:
        raise NotImplementedError

    def sync_spectral(self, f:'np.ndarray') -> None:
        raise NotImplementedError

    @property
    def n_dims(self) -> int:
        raise NotImplementedError

    @property
    def periodic_bounds(self) -> tuple:
        raise NotImplementedError

    @property
    def X(self) -> tuple:
        raise NotImplementedError