"""
Base class for domain decomposition.
"""
from abc import abstractmethod
import numpy as np
from numpy import ndarray


class DomainDecomposition:
    """
    Construct a grid of processors and decompose a global domain into subdomains.
    
    Description
    -----------
    Decompose the global domain into subdomains for parallel computing. The
    domain decomposition is done in a cartesian grid of processors. The
    decomposition can be done in multiple dimensions. Axes that are shared
    between processors can be specified (e.g. for fft)

    ::

                   ----------------------------------- 
                  /                /                /| 
                 /                /                / | 
                /                /                /  | 
               /                /                /   | 
              /                /                /    | 
             /                /                /    /| 
            /                /                /    / | 
            ----------------------------------    /  | 
           |                |                |   /   | 
           |   PROCESSOR    |   PROCESSOR    |  /    | 
           |     0, 1       |     1, 1       | /    /
           |                |                |/    /
           |----------------|----------------|    /     ^
           |                |                |   /     /
           |   PROCESSOR    |   PROCESSOR    |  /     / shared_axis
           |     0, 0       |     1, 0       | /     /
           |                |                |/
           ----------------------------------- 

    Parameters
    ----------
    `shape` : `tuple[int]`
        The total number of grid points in each dimension.
    `halo` : `int`, optional (default=0)
        The number of halo cells (ghost cells) around the local domain
        for the exchange of boundary values.
    `periods` : `tuple[bool]`, optional (default=None)
        A list of booleans indicating whether the domain is periodic in each dimension.
        If None, all dimensions are periodic.
    `shared_axes` : `list[int]`, optional (default=None)
        A list of axes that are shared between processors.
    `device_ids` : `list[int]`, optional (default=None)
        Optional list of device ids to use. If None, all devices are used.
        This option is useful for coupled simulations.
    """
    def __init__(self,
                 shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None):
        self._shape = shape
        self._n_dims = len(shape)
        self._halo = halo
        self._periods = periods or tuple(True for _ in range(self.n_dims))
        self._shared_axes = shared_axes
        self._rank = 0
        self._device_ids = device_ids
        self._p_dims = None

    # ================================================================
    #  Halo exchange
    # ================================================================

    @abstractmethod
    def sync(self, arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
        """
        Synchronize the halo regions of an array across all processes.

        Parameters
        ----------
        `arr` : ndarray
            The array to synchronize.
        `flat_axes` : list[int] | None
            Dimensions which are flat (no halo exchange). 
            If None, all dimensions are exchanged.
        """

    def sync_multiple(self, arr: list[ndarray]) -> list[ndarray]:
        """
        Synchronize the halo regions of multiple arrays across all processes.

        Parameters
        ----------
        `arr` : list[ndarray]
            The list of arrays to synchronize.
        """
        return [self.sync(a) for a in arr]

    # ================================================================
    #  Transpose
    # ================================================================

    @abstractmethod
    def transpose(self, arr: ndarray, axes_in: tuple[int], axes_out: tuple[int]) -> ndarray:
        """
        Transpose an array.

        Parameters
        ----------
        `arr` : ndarray
            The array to transpose.
        `axes_in` : tuple[int]
            The shared axes of the input.
        `axes_out` : tuple[int]
            The shared axes of the output.
        """

    # ================================================================
    #  Padding
    # ================================================================

    @abstractmethod
    def pad(self, arr: ndarray) -> ndarray:
        """
        Add padding to an array.

        Parameters
        ----------
        `arr` : ndarray
            The array to pad.
        """

    @abstractmethod
    def unpad(self, arr: ndarray) -> ndarray:
        """
        Remove padding from an array.

        Parameters
        ----------
        `arr` : ndarray
            The array to unpad.
        """

    # ================================================================
    #  Gather
    # ================================================================

    @abstractmethod
    def gather(self, 
               arr: ndarray, 
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None) -> ndarray:
        """
        Gather an array to a single process.

        Parameters
        ----------
        `arr` : ndarray
            The array to gather.
        `slc` : tuple[slice] (default=None)
            The slice of the array to gather.
            If None, gather the entire array.
        `dest_rank` : int (default=None)
            The rank of the process to gather to.
            If None, gather to all processes.
        """

    # ================================================================
    #  Array creation
    # ================================================================

    @abstractmethod
    def create_array(self, pad: bool = True) -> ndarray:
        """
        Create an array.

        Parameters
        ----------
        `pad` : bool
            Whether to add padding to the array.
        """

    @abstractmethod
    def create_meshgrid(self, *args: ndarray, pad: bool = True) -> tuple[ndarray]:
        """
        Create a meshgrid of arrays.

        Parameters
        ----------
        `args` : ndarray
            The arrays to meshgrid.
        `pad` : bool
            Whether to add padding to the meshgrid.
        """

    # ================================================================
    #  Array operations
    # ================================================================

    @abstractmethod
    def sum(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Sum an array across specified axes.

        Parameters
        ----------
        `arr` : ndarray
            The array to sum.
        `axes` : list[int] | None
            The axes to sum across. 
            If None, sum across all axes.
        """

    @abstractmethod
    def max(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Find the maximum value of an array across specified axes.

        Parameters
        ----------
        `arr` : ndarray
            The array to find the maximum value of.
        `axes` : list[int] | None
            The axes to find the maximum value across. 
            If None, find the maximum value across all axes.
        """

    @abstractmethod
    def min(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Find the minimum value of an array across specified axes.

        Parameters
        ----------
        `arr` : ndarray
            The array to find the minimum value of.
        `axes` : list[int] | None
            The axes to find the minimum value across. 
            If None, find the minimum value across all axes.
        """

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def n_dims(self) -> int:
        """
        Number of dimensions.
        """
        return self._n_dims

    @property
    def shape(self) -> tuple[int]:
        """
        Shape of the domain (number of grid points).
        """
        return self._shape

    @property
    def halo(self) -> int:
        """
        Width of the halo region (same for all dimensions).
        """
        return self._halo

    @property
    def periods(self) -> tuple[bool] | None:
        """
        Periodic boundaries of the domain.
        """
        return self._periods

    @property
    def rank(self) -> int:
        """
        Rank of the current process.
        """
        return self._rank

    @property
    def size(self) -> int:
        """
        Number of processes.
        """
        return np.prod(self.p_dims)

    @property
    def device_ids(self) -> list[int] | None:
        """
        List of device ids.
        """
        return self._device_ids

    @property
    def i_am_active(self) -> bool:
        """
        Whether the current process is active in this domain.
        """
        return self.rank in self.device_ids

    @property
    def p_dims(self) -> tuple[int]:
        """
        Number of processes in each dimension.
        """
        return self._p_dims

    @property
    def shared_axes(self) -> tuple[int]:
        """
        Axes shared by all processes.
        """
        return [i for i, x in enumerate(self.p_dims) if x == 1]


def get_domain_decomposition(backend: str) -> DomainDecomposition:
    """
    Get the domain decomposition class for the specified backend.

    Parameters
    ----------
    `backend` : str
        The backend to use. Options are 'single' and 'jax'.
    """
    fall_back = 'single'
    if backend == 'single':
        from .single_decomposition import SingleDecomposition
        return SingleDecomposition
    elif backend == 'jax':
        try:
            from .jax_decomposition import JaxDecomposition
            return JaxDecomposition
        except ImportError:
            print(f"Jax not available, falling back to {fall_back}")
            return get_domain_decomposition(fall_back)
    else:
        raise ValueError(f"Unknown backend: {backend}")