# Import external modules
from mpi4py import MPI
import numpy as np
# Import internal modules
from .subdomain import Subdomain

class DomainDecomposition:
    """
    Construct a grid of processors and decompose a global domain into subdomains.
    
    Description
    -----------
    Decompose the global domain into subdomains for parallel computing. The
    domain decomposition is done in a cartesian grid of processors. The
    decomposition can be done in multiple dimensions. Axes that are shared
    between processors can be specified (e.g. for fft)
    ```
    #          ----------------------------------- 
    #         /                /                /| 
    #        /                /                / | 
    #       /                /                /  | 
    #      /                /                /   | 
    #     /                /                /    | 
    #    /                /                /    /| 
    #   /                /                /    / | 
    #   ----------------------------------    /  | 
    #  |                |                |   /   | 
    #  |   PROCESSOR    |   PROCESSOR    |  /    | 
    #  |     0, 1       |     1, 1       | /    /
    #  |                |                |/    /
    #  |----------------|----------------|    /     ^
    #  |                |                |   /     /
    #  |   PROCESSOR    |   PROCESSOR    |  /     / shared_axis
    #  |     0, 0       |     1, 0       | /     /
    #  |                |                |/
    #  ----------------------------------- 
    ```

    Parameters
    ----------
    `n_global` : `tuple[int]`
        The total number of grid points in each dimension.
    `halo` : `int`, optional (default=0)
        The number of halo cells (ghost cells) around the local domain
        for the exchange of boundary values.
    `shared_axes` : `list[int]`, optional (default=None)
        A list of axes that are shared between processors.
    `reorder_comm` : `bool`, optional (default=True)
        Whether to reorder the communicator.
    
    Attributes
    ----------
    `n_dims` : `int`
        The number of dimensions.
    `n_global` : `tuple[int]`
        The total number of grid points in each dimension.
    `halo` : `int`
        The number of halo cells (ghost cells) around the local domain.
    `n_procs` : `tuple[int]`
        The number of processors in each direction.
    `shared_axes` : `list[int]`
        A list of axes that are shared between processors. A shared axis is an
        axis that only has one processor in that direction.
    `comm` : `MPI.Intracomm`
        The cartesian communicator that defines the processor grid.
    `size` : `int`
        The total number of processors.
    `rank` : `int`
        The rank of the current processor.
    `all_subdomains` : `list[Subdomain]`
        A list of all subdomains in the global domain.
    `my_subdomain` : `Subdomain`
        The local domain of the current processor.
    
    Methods
    -------
    `sync(arr: 'ndarray') -> None`
        Synchronize the halo regions of an array between neighboring domains.
    `sync_list(arrs: 'list[ndarray]') -> None`
        Synchronize the halo regions of a list of arrays.
    `sync_with_device() -> None`
        Synchronize the gpu device with the processor.
    `apply_boundary_condition(arr: 'ndarray', bc: 'ndarray', dimension, side)`
        Apply boundary condition to the halo regions of an array.
    
    Examples
    --------
    >>> from fridom.framework import config
    >>> from fridom.framework.domain_decomposition import DomainDecomposition
    >>> # create a domain decomposition that shares the x-axis
    >>> dom = DomainDecomposition(
            n_global=(128, 128), halo=2, shared_axes=[0])
    >>> 
    >>> # create a random array on the local domain
    >>> u = config.ncp.random.rand(*dom_x.my_subdomain.shape)
    >>>
    >>> # synchronize the halo regions between neighboring domains
    >>> dom_x.sync(u)
    """
    def __init__(
            self, 
            n_global: 'tuple[int]', 
            halo: int = 0,
            shared_axes: 'list[int] | None' = None, 
            reorder_comm = True,
            ) -> None: ...

    def sync(self, arr: np.ndarray, flat_axes: list | None = None) -> None:
        """
        # Synchronize Halos
        Synchronize the halo regions of an array between neighboring domains.
        
        Description
        -----------
        This function synchronizes the halo regions of an array between neighboring
        domains. The synchronization is done in place (no return value). 
        Synchronization is always periodic in all directions.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The array to synchronize.
        `flat_axes` : `list[int]`, optional (default=None)
            A list of axes to skip synchronization.
        
        Examples
        --------
        >>> from fridom.framework import config
        >>> from fridom.framework.domain_decomposition import DomainDecomposition
        >>> # create a domain decomposition
        >>> domain = DomainDecomposition(n_global=(128, 128), shared_axes=[0])
        >>> # create a random array on the local domain
        >>> u = config.ncp.random.rand(domain.my_subdomain.shape)
        >>> # synchronize the halo regions between neighboring domains
        >>> domain.sync(u)
        """

    def sync_list(self, 
                  arrs: 'list[np.ndarray]', 
                  flat_axes: list | None = None) -> None:
        """
        Synchronize the halo regions of a list of arrays between neighboring domains.
        
        Description
        -----------
        This function synchronizes the halo regions of a list of arrays between
        neighboring domains. The synchronization is done in place (no return value).
        Synchronization is always periodic in all directions.
        
        Parameters
        ----------
        `arrs` : `list[ndarray]`
            List of arrays to synchronize.
        `flat_axes` : `list[int]`, optional (default=None)
            A list of axes to skip synchronization.
        
        Examples
        --------
        >>> from fridom.framework import config
        >>> from fridom.framework.domain_decomposition import DomainDecomposition
        >>> # create a domain decomposition
        >>> domain = DomainDecomposition(n_global=(128, 128), shared_axes=[0])
        >>> # create a random array on the local domain
        >>> u = config.ncp.random.rand(domain.my_subdomain.shape)
        >>> v = config.ncp.random.rand(domain.my_subdomain.shape)
        >>> # synchronize the halo regions between neighboring domains
        >>> domain.sync_list([u, v])
        """

    def sync_with_device(self):
        """
        Synchronize the gpu device with the processor.
        
        Description
        -----------
        When using the cupy backend, and calling a cupy function, the cpu does
        not wait for the gpu to finish the computation. This can lead to the cpu
        and gpu being out of sync. This function ensures that the cpu waits for
        all gpu computations to finish.
        """

    def apply_boundary_condition(
            self, 
            arr: np.ndarray, 
            bc: np.ndarray, 
            axis: int, 
            side: str
            ) -> None:
        """
        Apply boundary condition to the halo regions of an array.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array to apply the boundary condition to.
        `bc` : `np.ndarray`
            The boundary condition to apply.
        `axis` : `int`
            The dimension to apply the boundary condition to.
        `side` : `str`
            The side to apply the boundary condition to. left or right.
        
        Raises
        ------
        `ValueError`
            f `side` is not either 'left' or 'right'.
        """

    @property
    def n_dims(self) -> int:
        """The number of dimensions."""

    @property
    def n_global(self) -> tuple[int]:
        """The total number of grid points in each dimension."""

    @property
    def halo(self) -> int:
        """The number of halo cells (ghost cells) around the local domain."""

    @property
    def n_procs(self) -> tuple[int]:
        """The number of processors in each direction."""

    @property
    def shared_axes(self) -> list[int]:
        """A list of axes that are shared between processors."""

    @property
    def comm(self) -> MPI.Intracomm:
        """The cartesian communicator that defines the processor grid."""

    @property
    def size(self) -> int:
        """The total number of processors."""

    @property
    def rank(self) -> int:
        """The rank of the current processor."""

    @property
    def all_subdomains(self) -> list[Subdomain]:
        """A list of all subdomains in the global domain."""

    @property
    def my_subdomain(self) -> Subdomain:
        """The local domain of the current processor."""