# Import external modules
from mpi4py import MPI

class Subdomain:
    """
    A class that holds information about a local domain in the global processor
    grid.
    
    Description
    -----------
    A subdomain represents a local domain in the global grid of processors. It
    holds information about the position in the processor grid, as well as the
    position in the global index space. The subdomain also provides methods to
    convert slices between the global and local index space.

    Parameters
    ----------
    `rank` : `int`
        The global rank of the processor.
    `comm` : `MPI.Intracomm`
        The cartesian communicator that defines the processor grid.
    `n_global` : `list[int]`
        The global number of grid points in each dimension.
    `halo` : `int`, optional (default: 0)
        The number of halo cells around the local domain for the exchange
        of boundary values.
    
    Attributes
    ----------
    `n_global` : `list[int]`
        The global number of grid points in each dimension.
    `halo` : `int`
        The number of halo cells around the local domain for the exchange of
        boundary values.
    `rank` : `int`
        The global rank of the subdomains processor in the communicator.
    `coord` : `list[int]`
        The coordinates of the processor in the processor grid.
    `is_left_edge` : `list[bool]`
        A list of booleans that indicate if the processor is at the left edge
        of the global domain.
    `is_right_edge` : `list[bool]`
        A list of booleans that indicate if the processor is at the right edge
        of the global domain.
    `shape` : `tuple[int]`
        The number of grid points in the local domain including halo.
    `inner_shape` : `tuple[int]`
        The number of grid points in the local domain without halo.
    `position` : `tuple[int]`
        The position of the local domain in the global grid (index coordinates).
    `global_slice` : `tuple[slice]`
        The slice of the global domain that corresponds to the local domain.
    `inner_slice` : `tuple[slice]`
        The slice of the local domain without halo.
    
    Methods
    -------
    `has_overlap(other: 'Subdomain') -> bool`
        Check if the local domain overlaps with another domain.
    `get_overlap_slice(other: 'Subdomain') -> 'tuple[slice]'`
        Get the slice of the local domain that overlaps with another domain.
    `g2l_slice(global_slice: 'tuple[slice]') -> 'tuple[slice]'`
        Convert a global slice to a local slice.
    `l2g_slice(local_slice: 'tuple[slice]') -> 'tuple[slice]'`
        Convert a local slice to a global slice.
    
    Examples
    --------
    The following example must be run with MPI enabled and with 6 processors:

    .. code-block:: bash

        mpirun -n 6 python example.py

    with the following content in `example.py`:

    .. code-block:: python

        from mpi4py import MPI
        from fridom.framework.domain_decomposition import Subdomain
        # Create a cartesian communicator with a processor grid of 3x2 processors
        cart_comm = MPI.COMM_WORLD.Create_cart(dims=[3,2], periods=[True,True])
        rank = cart_comm.Get_rank()

        # Create a subdomain for the processor of the current rank
        subdomain = Subdomain(rank=rank, comm=cart_comm, n_global=[128,128], halo=1)

        # Print the attributes of the subdomain for the processor with rank 1
        if rank == 1:
            print(subdomain.coord)         # [0, 1]
            print(subdomain.inner_shape)   # [42, 64]
            print(subdomain.shape)         # [44, 66] -> +2 because of the halo
            print(subdomain.position)      # [0, 64]
            print(subdomain.global_slice)  # (slice(0, 42, None), slice(64, 128, None))

        # Create a numpy array with the shape of the local domain
        import numpy as np
        data = np.zeros(subdomain.shape)

        # set the values in the area [25:50, 20:120] to 1
        local_slice = subdomain.g2l_slice((slice(25,50), slice(20,120)))
        data[local_slice] = 1
    """
    def __init__(self, rank: int, 
                 comm: MPI.Cartcomm,
                 n_global: 'list[int]',
                 halo: int = 0,
                 ) -> None:...

    def has_overlap(self, other: 'Subdomain') -> bool:
        """
        Check if the local domain overlaps with another domain.
        
        Description
        -----------
        The function checks if the global_slice of this subdomain has an overlap
        with the global_slice of another subdomain. When two subdomains are 
        constructed from the same global domain, they should never overlap.
        However, when two subdomains are constructed from different global domains,
        they might overlap.
        
        Parameters
        ----------
        `other` : `Subdomain`
            The other subdomain to check for overlap.
        
        Returns
        -------
        `bool`
            True if the subdomains overlap, False otherwise.
        
        Examples
        --------
        The following example must be run with MPI enabled and with 2 processors:

        .. code-block:: bash

            mpirun -n 2 python example.py

        with the following content in `example.py`:

        .. code-block:: python

            from mpi4py import MPI
            from fridom.framework.domain_decomposition import Subdomain

            # Create a cartesian communicator with a processor grid of 2x1 processors
            cart_comm = MPI.COMM_WORLD.Create_cart(dims=[2,1], periods=[True,True])

            # Create two subdomains for the processors with rank 0 and 1
            subdomain0 = Subdomain(
                rank=0, comm=cart_comm, n_global=[128,128], halo=1)
            subdomain1 = Subdomain(
                rank=1, comm=cart_comm, n_global=[128,128], halo=1)
            print(subdomain0.has_overlap(subdomain1))  # False

            # Create a cartesian communicator with a processor grid of 1x2 processors
            cart_comm = MPI.COMM_WORLD.Create_cart(dims=[1,2], periods=[True,True])

            # Create a subdomain for the processor in the new communicator
            subdomain2 = Subdomain(
                rank=0, comm=cart_comm, n_global=[128,128], halo=1)
            print(subdomain0.has_overlap(subdomain2))  # True
        """

    def get_overlap_slice(self, other: 'Subdomain') -> 'tuple[slice]':
        """
        Get the slice of the local domain that overlaps with another domain.
        
        Description
        -----------
        The function returns the slice of the local domain that overlaps with
        another domain. The slice is given in local coordinates and includes the
        halo cells. If the domains do not overlap, the function returns slices
        that won't select any elements, e.g., `slice(1,0,None)`, where the
        start index is greater than the stop index.
        
        Parameters
        ----------
        `other` : `Subdomain`
            The other subdomain to get the overlap with.
        
        Returns
        -------
        `tuple[slice]`
            The slice of the local domain that overlaps with the other domain.
        
        Examples
        --------
        The following example must be run with MPI enabled and with 2 processors:

        .. code-block:: bash

            mpirun -n 2 python example.py

        with the following content in `example.py`:

        .. code-block:: python

            from mpi4py import MPI
            from fridom.framework.domain_decomposition import Subdomain

            # Create a 2x1 and 1x2 processor grid:
            cart_comm1 = MPI.COMM_WORLD.Create_cart(dims=[2,1], periods=[True,True])
            cart_comm2 = MPI.COMM_WORLD.Create_cart(dims=[1,2], periods=[True,True])

            # Create two subdomains for the processors with rank 0 and 1
            subdomain1 = Subdomain(
                rank=0, comm=cart_comm1, n_global=[128,128], halo=1)
            subdomain2 = Subdomain(
                rank=0, comm=cart_comm2, n_global=[128,128], halo=1)

            print(subdomain1.global_slice)  # [0:64, 0:128]
            print(subdomain2.global_slice)  # [0:128, 0:64]

            overlap = subdomain1.get_overlap_slice(subdomain2)
            print(overlap)  # [1:65, 1:65]  (+1 because of the halo region)
        """

    def g2l_slice(self, global_slice: 'tuple[slice]') -> 'tuple[slice]':
        """
        Convert a slice from the global space to the local space.
        
        Parameters
        ----------
        `global_slice` : `tuple[slice]`
            A tuple of slices in the global index space.
        
        Returns
        -------
        `tuple[slice]`
            A tuple of slices in the local index space.
        
        Examples
        --------
        The following example must be run with MPI enabled and with 6 processors:

        .. code-block:: bash

            mpirun -n 6 python example.py

        with the following content in `example.py`:

        .. code-block:: python

            from mpi4py import MPI
            from fridom.framework.domain_decomposition import Subdomain

            # Create a cartesian communicator with a processor grid of 3x2 processors
            cart_comm = MPI.COMM_WORLD.Create_cart(dims=[3,2], periods=[True,True])
            rank = cart_comm.Get_rank()

            # Create a subdomain for the processor of the current rank
            subdomain = Subdomain(
                rank=rank, comm=cart_comm, n_global=[128,128], halo=1)

            # Create a numpy array with the shape of the local domain
            import numpy as np
            data = np.zeros(subdomain.shape)

            # set the values in the area [25:50, 20:120] to 1
            global_slice = (slice(25,50), slice(20,120))
            local_slice = subdomain.g2l_slice(global_slice)
            data[local_slice] = 1
        """

    def l2g_slice(self, local_slice: tuple[slice]) -> tuple[slice]:
        """
        Convert a slice from the local space to the global space.
        
        Parameters
        ----------
        `local_slice` : `tuple[slice]`
            The slice in the local index space.
        
        Returns
        -------
        `tuple[slice]`
            The slice in the global index space.
        
        Examples
        --------
        The following example must be run with MPI enabled and with 6 processors:

        .. code-block:: bash

            mpirun -n 6 python example.py

        with the following content in `example.py`:

        .. code-block:: python

            from mpi4py import MPI
            from fridom.framework.domain_decomposition import Subdomain

            # Create a cartesian communicator with a processor grid of 3x2 processors
            cart_comm = MPI.COMM_WORLD.Create_cart(dims=[3,2], periods=[True,True])
            rank = cart_comm.Get_rank()

            # Create a subdomain for the processor of the current rank
            subdomain = Subdomain(
                rank=rank, comm=cart_comm, n_global=[128,128], halo=1)

            # Create a numpy array
            import numpy as np
            u_global = np.random.rand(*subdomain.n_global)
            u_local = np.zeros(subdomain.shape)
            u_local[subdomain.inner_slice] = u_global[subdomain.global_slice]

            # Create a local slice and convert it to a global slice
            local_slice = (slice(10,20), slice(20,30))
            global_slice = subdomain.l2g_slice(local_slice)

            # Check if the values are the same
            assert np.allclose(u_global[global_slice], u_local[local_slice])
        """

    @property
    def n_global(self) -> 'list[int]':
        """The global number of grid points in each dimension."""

    @property
    def halo(self) -> int:
        """The number of halo cells around the local domain for the exchange 
        of boundary values."""

    @property
    def rank(self) -> int:
        """The global rank of the subdomains processor in the communicator."""

    @property
    def coord(self) -> 'list[int]':
        """The coordinates of the processor in the processor grid."""

    @property
    def is_left_edge(self) -> 'list[bool]':
        """A list of booleans that indicate if the processor is at the left edge
        of the global domain."""

    @property
    def is_right_edge(self) -> 'list[bool]':
        """A list of booleans that indicate if the processor is at the right edge
        of the global domain."""

    @property
    def shape(self) -> 'tuple[int]':
        """The number of grid points in the local domain including halo."""

    @property
    def inner_shape(self) -> 'tuple[int]':
        """The number of grid points in the local domain without halo."""

    @property
    def position(self) -> 'tuple[int]':
        """The position of the local domain in the global grid (index 
        coordinates)."""

    @property
    def global_slice(self) -> 'tuple[slice]':
        """The slice of the global domain that corresponds to the local domain."""
    
    @property
    def inner_slice(self) -> 'tuple[slice]':
        """The slice of the local domain without halo."""


