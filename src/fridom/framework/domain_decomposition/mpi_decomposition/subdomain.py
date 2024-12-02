from fridom.framework import utils

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
    `n_global` : `tuple[int]`
        The global number of grid points in each dimension.
    `halo` : `int`, optional (default: 0)
        The number of halo cells around the local domain for the exchange
        of boundary values.
    
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
        subdomain = Subdomain(rank=rank, comm=cart_comm, n_global=(128,128), halo=1)

        # Print the attributes of the subdomain for the processor with rank 1
        if rank == 1:
            print(subdomain.coord)         # (0, 1)
            print(subdomain.inner_shape)   # (42, 64)
            print(subdomain.shape)         # (44, 66) -> +2 because of the halo
            print(subdomain.position)      # (0, 64)
            print(subdomain.global_slice)  # (slice(0, 42, None), slice(64, 128, None))

        # Create a numpy array with the shape of the local domain
        import numpy as np
        data = np.zeros(subdomain.shape)

        # set the values in the area [25:50, 20:120] to 1
        local_slice = subdomain.g2l_slice((slice(25,50), slice(20,120)))
        data[local_slice] = 1
    """
    def __init__(self, rank: int, 
                 comm,
                 n_global: 'tuple[int]',
                 halo: int = 0,
                 ) -> None:
        # get the processor coordinates and dimensions of the processor grid
        n_dims = len(n_global)
        if utils.MPI_AVAILABLE:
            coord = tuple(comm.Get_coords(rank))  # processor coordinates
            n_procs = tuple(comm.Get_topo()[0])   # number of processors in each dim.
        else:
            coord = (0,) * n_dims
            n_procs = (1,) * n_dims

        # check if a processor is at the edge of the global domain
        is_left_edge = tuple(c == 0 for c in coord)
        is_right_edge = tuple(c == n-1 for c,n in zip(coord, n_procs))

        # get the number of grid points in the local domain (the inner shape)
        # we decompose the number of grid points in the local domain into the
        # base number of grid points that each processor gets and the remainder.
        # The remainder is added to the last processor such that the global
        # number of grid points is preserved.
        # Consider for example the case of 102 grid points and 10 processors:
        # Each processor gets 10 grid points, except the last one, which gets 12.
        # Hence the base number of grid points is 10 and the remainder is 0 for
        # all processors except the last one, where it is 2.

        # number of local elements in each dimension
        n_base = [n_grid // n_proc for n_grid, n_proc in zip(n_global, n_procs)]
        remainder = [n_grid % n_proc if (c == n_proc - 1) else 0
                     for c, n_grid, n_proc in zip(coord, n_global, n_procs)]
        inner_shape = tuple(n+r for n,r in zip(n_base, remainder))
        shape = tuple(n + 2*halo for n in inner_shape)
        inner_slice = tuple([slice(None)] * n_dims)
        if halo > 0:
            inner_slice = tuple([slice(halo, -halo)] * n_dims)

        # get the start position of the local domain in the global grid
        position = tuple(c * n for c,n in zip(coord, n_base))
        global_slice = tuple(slice(p,p+s) for p,s in zip(position, inner_shape))

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        self._n_global: list[int]        = n_global
        self._halo: int                  = halo
        self._rank: int                  = rank
        self._coord: tuple[int]          = coord
        self._is_left_edge: tuple[bool]  = is_left_edge
        self._is_right_edge: tuple[bool] = is_right_edge
        self._shape: tuple[int, ...]     = shape
        self._inner_shape: tuple[int, ...]    = inner_shape
        self._position: tuple[int, ...]       = position
        self._global_slice: tuple[slice, ...] = global_slice
        self._inner_slice: tuple[slice, ...]  = inner_slice
        
        return

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
                rank=0, comm=cart_comm, n_global=(128,128), halo=1)
            subdomain1 = Subdomain(
                rank=1, comm=cart_comm, n_global=(128,128), halo=1)
            print(subdomain0.has_overlap(subdomain1))  # False

            # Create a cartesian communicator with a processor grid of 1x2 processors
            cart_comm = MPI.COMM_WORLD.Create_cart(dims=[1,2], periods=[True,True])

            # Create a subdomain for the processor in the new communicator
            subdomain2 = Subdomain(
                rank=0, comm=cart_comm, n_global=(128,128), halo=1)
            print(subdomain0.has_overlap(subdomain2))  # True
        """
        # if there is an overlap, than every dimension must have an overlap
        for me, you in zip(self.global_slice, other.global_slice):
            if me.start >= you.stop or you.start >= me.stop:
                return False

        # if we reach this point, there is an overlap
        return True

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
                rank=0, comm=cart_comm1, n_global=(128,128), halo=1)
            subdomain2 = Subdomain(
                rank=0, comm=cart_comm2, n_global=(128,128), halo=1)

            print(subdomain1.global_slice)  # [0:64, 0:128]
            print(subdomain2.global_slice)  # [0:128, 0:64]

            overlap = subdomain1.get_overlap_slice(subdomain2)
            print(overlap)  # [1:65, 1:65]  (+1 because of the halo region)
        """
        # first get the overlap in the global coordinates
        global_overlap = []
        for me, you in zip(self.global_slice, other.global_slice):
            start = max(me.start, you.start)
            stop = min(me.stop, you.stop)
            global_overlap.append(slice(start, stop))

        # convert the global overlap to local coordinates
        return self.g2l_slice(tuple(global_overlap))

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
                rank=rank, comm=cart_comm, n_global=(128,128), halo=1)

            # Create a numpy array with the shape of the local domain
            import numpy as np
            data = np.zeros(subdomain.shape)

            # set the values in the area [25:50, 20:120] to 1
            global_slice = (slice(25,50), slice(20,120))
            local_slice = subdomain.g2l_slice(global_slice)
            data[local_slice] = 1
        """
        local_slice = []
        for g, p, s in zip(global_slice, self.position, self.shape):
            start = g.start - p + self.halo
            stop = g.stop - p + self.halo
            local_slice.append(slice(start, stop))
        return tuple(local_slice)

    def l2g_slice(self, local_slice: 'tuple[slice]') -> 'tuple[slice]':
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
                rank=rank, comm=cart_comm, n_global=(128,128), halo=1)

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
        global_slice = []
        for l, p, s in zip(local_slice, self.position, self.shape):
            start = l.start + p - self.halo
            stop = l.stop + p - self.halo
            global_slice.append(slice(start, stop))
        return tuple(global_slice)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def n_global(self) -> 'tuple[int]':
        """The global number of grid points in each dimension."""
        return self._n_global

    @property
    def halo(self) -> int:
        """The number of halo cells around the local domain for the exchange 
        of boundary values."""
        return self._halo

    @property
    def rank(self) -> int:
        """The global rank of the subdomains processor in the communicator."""
        return self._rank

    @property
    def coord(self) -> 'tuple[int]':
        """The coordinates of the processor in the processor grid."""
        return self._coord

    @property
    def is_left_edge(self) -> 'tuple[bool]':
        """A list of booleans that indicate if the processor is at the left edge
        of the global domain."""
        return self._is_left_edge

    @property
    def is_right_edge(self) -> 'tuple[bool]':
        """A list of booleans that indicate if the processor is at the right edge
        of the global domain."""
        return self._is_right_edge

    @property
    def shape(self) -> 'tuple[int]':
        """The number of grid points in the local domain including halo."""
        return self._shape

    @property
    def inner_shape(self) -> 'tuple[int]':
        """The number of grid points in the local domain without halo."""
        return self._inner_shape

    @property
    def position(self) -> 'tuple[int]':
        """The position of the local domain in the global grid (index 
        coordinates)."""
        return self._position

    @property
    def global_slice(self) -> 'tuple[slice]':
        """The slice of the global domain that corresponds to the local domain."""
        return self._global_slice
    
    @property
    def inner_slice(self) -> 'tuple[slice]':
        """The slice of the local domain without halo."""
        return self._inner_slice