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
    """
    def __init__(self) -> None:
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