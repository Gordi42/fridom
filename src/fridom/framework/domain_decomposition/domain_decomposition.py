"""Base class for domain decomposition."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy import ndarray

import fridom.framework as fr


@fr.utils.jaxify
class DomainDecomposition:

    """
    Construct a processor grid and decompose a domain into subdomains.

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
    shape : tuple[int]
        The total number of grid points in each dimension.
    halo : int, optional
        The number of halo cells (ghost cells) around the local domain
        for the exchange of boundary values (default: 0).
    periods : tuple[bool], optional
        A list of booleans indicating whether the domain is periodic in
        each dimension. If None, all dimensions are periodic (default: None).
    shared_axes : list[int], optional
        A list of axes that are shared between processors (default: None).
    device_ids : list[int], optional
        Optional list of device ids to use. If None, all devices are used.
        This option is useful for coupled simulations (default: None).
    """

    def __init__(self,
                 shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None) -> None:
        self._shape = shape
        self._n_dims = len(shape)
        self._halo = halo
        self._periods = periods or tuple(True for _ in range(self.n_dims))
        self._shared_axes = shared_axes or []
        self._rank = 0
        self._device_ids = device_ids
        self._p_dims = None

    # ================================================================
    #  Halo exchange
    # ================================================================

    @abstractmethod
    def sync(
        self, arr: ndarray, flat_axes: list[int] | None = None,
    ) -> ndarray:
        """
        Synchronize the halo regions of an array across all processes.

        Parameters
        ----------
        arr : ndarray
            The array to synchronize.
        flat_axes : list[int] | None
            Dimensions which are flat (no halo exchange).
            If None, all dimensions are exchanged.
        """

    def sync_multiple(self, arr: list[ndarray]) -> list[ndarray]:
        """
        Synchronize the halo regions of multiple arrays across all processes.

        Parameters
        ----------
        arr : list[ndarray]
            The list of arrays to synchronize.
        """
        return [self.sync(a) for a in arr]

    # ================================================================
    #  Apply Transform (e.g. FFT)
    # ================================================================

    def parallel_forward_transform(self, func: callable) -> callable:
        """
        Parallel forward transform.

        Parameters
        ----------
        func : callable
            The function to apply the forward transform to.
            func(arr: ndarray, axes: list[int] | None = None) -> ndarray
        """
        def wrapper(arr: ndarray, axes: list[int] | None = None) -> ndarray:
            # unpad the array
            arr = self.unpad(arr)
            # apply the forward transform
            return func(arr, axes=axes)
        return wrapper

    def parallel_backward_transform(self, func: callable) -> callable:
        """
        Parallel backward transform.

        Parameters
        ----------
        func : callable
            The function to apply the backward transform to.
            func(arr: ndarray, axes: list[int] | None = None) -> ndarray
        """
        def wrapper(arr: ndarray, axes: list[int] | None = None) -> ndarray:
            # apply the backward transform
            arr = func(arr, axes=axes)
            # pad the array
            return self.pad(arr)
        return wrapper

    # ================================================================
    #  Padding
    # ================================================================

    @abstractmethod
    def pad(self, arr: ndarray) -> ndarray:
        """
        Add padding to an array.

        Parameters
        ----------
        arr : ndarray
            The array to pad.
        """

    @abstractmethod
    def unpad(self, arr: ndarray) -> ndarray:
        """
        Remove padding from an array.

        Parameters
        ----------
        arr : ndarray
            The array to unpad.
        """

    # ----------------------------------------------------------------
    #  Spectral paddings
    # ----------------------------------------------------------------

    def pad_extend(self, arr: ndarray) -> ndarray:
        """
        Extend the array with zeros (for spectral padding).

        Parameters
        ----------
        arr : ndarray
            The array to pad.

        Returns
        -------
        ndarray
            The padded array.
        """

    def unpad_extend(self, arr: ndarray) -> ndarray:
        """
        Remove the extension of the array (for spectral padding).

        Parameters
        ----------
        arr : ndarray
            The array to unpad.

        Returns
        -------
        ndarray
            The unpadded array.
        """

    def pad_trim(self, arr: ndarray) -> ndarray:
        """
        Set the padded region to zero (for spectral padding).

        Parameters
        ----------
        arr : ndarray
            The array to pad.
        """

    # ================================================================
    #  Gather
    # ================================================================

    @abstractmethod
    def gather(self,
               arr: ndarray,
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,
               spectral: bool = False) -> ndarray:
        """
        Gather an array to a single process.

        Parameters
        ----------
        arr : ndarray
            The array to gather.
        slc : tuple[slice], optional
            The slice of the array to gather.
            If None, gather the entire array (default: None).
        dest_rank : int, optional
            The rank of the process to gather to.
            If None, gather to all processes (default: None).
        spectral : bool
            Whether the array is in spectral space.
        """

    # ================================================================
    #  Array creation
    # ================================================================

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def create_meshgrid(self,
                        *args: ndarray,
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:
        """
        Create a meshgrid of arrays.

        Parameters
        ----------
        args : ndarray
            The arrays to meshgrid.
        pad : bool
            Whether to add padding to the meshgrid.
        spectral : bool
            Whether the meshgrid is in spectral space.
        """

    # ================================================================
    #  Array operations
    # ================================================================

    @abstractmethod
    def sum(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        """
        Sum an array across specified axes.

        Parameters
        ----------
        arr : ndarray
            The array to sum.
        axes : list[int] | None
            The axes to sum across.
            If None, sum across all axes.
        spectral : bool
            Whether the array is in spectral space.
        """

    @abstractmethod
    def max(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        """
        Find the maximum value of an array across specified axes.

        Parameters
        ----------
        arr : ndarray
            The array to find the maximum value of.
        axes : list[int] | None
            The axes to find the maximum value across.
            If None, find the maximum value across all axes.
        spectral : bool
            Whether the array is in spectral space.
        """

    @abstractmethod
    def min(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        """
        Find the minimum value of an array across specified axes.

        Parameters
        ----------
        arr : ndarray
            The array to find the minimum value of.
        axes : list[int] | None
            The axes to find the minimum value across.
            If None, find the minimum value across all axes.
        spectral : bool
            Whether the array is in spectral space.
        """

    @abstractmethod
    def cumsum(self,
               arr: ndarray,
               axis: int) -> ndarray:
        """
        Cumulative sum of an array along a specified axis.

        Parameters
        ----------
        arr : ndarray
            The array to cumsum.
        axis : int
            The axis to cumsum along.

        """

    @abstractmethod
    def inv_cumsum(self,
                   arr: ndarray,
                   axis: int) -> ndarray:
        """
        Inverse cumulative sum of an array along a specified axis.

        Parameters
        ----------
        arr : ndarray
            The array to inv_cumsum.
        axis : int
            The axis to inv_cumsum along.

        """

    @abstractmethod
    def roll(self,
             arr: ndarray,
             shift: int | tuple[int],
             axis: int | tuple[int]) -> ndarray:
        """
        Roll an array along specified axes.

        Parameters
        ----------
        arr : ndarray
            The array to roll.
        shift : int | tuple[int]
            The number of places by which elements are shifted. Rolling
            the array [1,2,3,4,5] with shift=1 results in [5,1,2,3,4].
            If a tuple is given, the axis must also be a tuple of the
            same length.
        axis : int | tuple[int]
            The axis or axes to roll along. If a tuple is given, the shift must
            also be a tuple of the same length.

        """


    # ================================================================
    #  Helper functions
    # ================================================================
    def shard_map(self, func: callable) -> callable:
        """
        Decorate a function to apply it to the active processes only.

        Parameters
        ----------
        func : callable
            The function to apply.
        """
        return func

    def to_alterative_sharding(self, arr: ndarray) -> ndarray:
        """
        Convert an array to the alternative sharding.

        Parameters
        ----------
        arr : ndarray
            The array to convert.

        """
        return arr

    def to_main_sharding(self, arr: ndarray) -> ndarray:
        """
        Convert an array to the main sharding.

        Parameters
        ----------
        arr : ndarray
            The array to convert.

        """
        return arr

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self._n_dims

    @property
    def shape(self) -> tuple[int]:
        """Shape of the domain (number of grid points)."""
        return self._shape

    @property
    def halo(self) -> int:
        """Width of the halo region (same for all dimensions)."""
        return self._halo

    @property
    def periods(self) -> tuple[bool] | None:
        """Periodic boundaries of the domain."""
        return self._periods

    @property
    def parallel(self) -> bool:
        """Whether the domain is parallel."""
        return self.size > 1

    @property
    def rank(self) -> int:
        """Rank of the current process."""
        return self._rank

    @property
    def size(self) -> int:
        """Number of processes."""
        return np.prod(self.p_dims)

    @property
    def device_ids(self) -> list[int] | None:
        """List of device ids."""
        return self._device_ids

    @property
    def i_am_active(self) -> bool:
        """Whether the current process is active in this domain."""
        return self.rank in self.device_ids

    @property
    def p_dims(self) -> tuple[int]:
        """Number of processes in each dimension."""
        return self._p_dims

    @property
    def shared_axes(self) -> tuple[int]:
        """Axes shared by all processes."""
        return [i for i, x in enumerate(self.p_dims) if x == 1]


def get_default_domain_decomposition() -> DomainDecomposition:
    """
    Get the default domain decomposition class.

    Description
    -----------
    If more than one device is available, the domain is decomposed across
    the devices using the JaxDecomposition. Otherwise, the
    SingleDecomposition is used.
    """
    import jax  # noqa: PLC0415 (avoid initializing the backend on import)
    # if we only have one available device, we use single decomposition
    if jax.device_count() == 1:
        return fr.domain_decomposition.SingleDecomposition
    # otherwise, we use the jax decomposition
    return fr.domain_decomposition.JaxDecomposition
