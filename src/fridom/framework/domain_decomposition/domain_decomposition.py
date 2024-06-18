from mpi4py import MPI
from .subdomain import Subdomain

def make_slice_list(s: slice, n_dims: int):
    """
    # Make Slice List
    Create a list of slices for halo exchange.
        
    Description
    -----------
    Lets assume we want to access the halo region that is shared with the
    previous processor. For a 2D domain, two such regions exist, one in the
    x-direction and one in the y-direction. Hence we need to create two
    slices, one for each dimension. In this example and a halo of 2, the
    slices would be (slice(0, 2), slice(None)) and 
    (slice(None), slice(0, 2)). This function for s=slice(0, 2) would
    return the list of the two slice-tuples.
        
    Parameters
    ----------
    `s` : `slice`
        A 1D slice that defines the halo region in one dimension.
    `n_dims` : `int`
        The number of dimensions.
        
    Returns
    -------
    `list[tuple[slice]]`
        A list of slice tuples for each dimension.
        
    Examples
    --------
    ```
    halo = 2
    n_dims = 2
    send_to_next = make_slice_list(slice(-2*halo, -halo))
    send_to_next
    # [(slice(-4, -2), slice(None)), (slice(None), slice(-4, -2))]
    # send_to_next[0] is used to send data to the next processor in the
    # first dimension, send_to_next[1] is used to send data to the next
    # processor in the second dimension
    ```
    """
    slice_list = []
    for i in range(n_dims):
        full_slice = [slice(None)]*n_dims
        full_slice[i] = s
        slice_list.append(tuple(full_slice))
    return slice_list

def set_device(backend_is_cupy):
    """
    # Set GPU Device
    Sync the gpu device with the processor rank. This function should be
    called once at the beginning of the simulation.
        
    Description
    -----------
    Let's assume we have a node with 4 GPUs and start a program with 4
    processors. By default, each processor will uses the first GPU. This
    function ensures that each processor uses a different GPU.
        
    Parameters
    ----------
    `backend_is_cupy` : `bool`
        Whether the backend is cupy or not.
        
    Returns
    -------
    `None`
    """
    if backend_is_cupy:
        import cupy as cp
        comm_node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        node_rank = comm_node.Get_rank()
        num_gpus = cp.cuda.runtime.getDeviceCount()
        device_id = node_rank % num_gpus
        cp.cuda.Device(device_id).use()
    return

class DomainDecomposition:
    """
    # Domain Decomposition
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
    
    Attributes
    ----------
    `n_dims` : `int`
        The number of dimensions.
    `n_global` : `list[int]`
        The total number of grid points in each dimension.
    `halo` : `int`
        The number of halo cells (ghost cells) around the local domain.
    `backend` : `str`
        The backend to use for the computation (numpy or cupy).
    `ncp` : `module`
        The numpy or cupy module.
    `n_procs` : `list[int]`
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
    `__init__(n_global: 'list[int]', **kwargs) -> None`
        Initialize the domain decomposition.
    `sync(arr: 'ndarray') -> None`
        Synchronize the halo regions of an array between neighboring domains.
    `sync_list(arr_list: 'list[ndarray]') -> None`
        Synchronize the halo regions of a list of arrays.
    `sync_with_device() -> None`
        Synchronize the gpu device with the processor.
    
    Examples
    --------
    ```
    # create two domains where one shares the x-axis and the other the y-axis
    dom = DomainDecomposition(n_global=[128]*2, halo=2, shared_axes=[0])

    # create a random array on the local domain
    u = dom_x.ncp.random.rand(*dom_x.my_subdomain.shape)

    # synchronize the halo regions between neighboring domains
    dom_x.sync(u)
    ```
    """
    def __init__(self, 
                 n_global: 'list[int]', 
                 halo: int = 0,
                 shared_axes: 'list[int]' = None,
                 reorder_comm = True,
                 backend = "numpy", # or "cupy"
                 ) -> None:
        """
        # Initialize the domain decomposition.
        Create a grid of processors and decompose a global domain into subdomains.
        
        Parameters
        ----------
        `n_global` : `list[int]`
            The total number of grid points in each dimension.
        `halo` : `int`, optional (default=0)
            The number of halo cells (ghost cells) around the local domain
            for the exchange of boundary values.
        `shared_axes` : `list[int]`, optional (default=None)
            A list of axes that are shared between processors.
        `reorder_comm` : `bool`, optional (default=True)
            Whether to reorder the communicator.
        `backend` : `str`, optional (default="numpy")
            The backend to use for the computation (numpy or cupy).
        
        Returns
        -------
        `None`
        
        Raises
        ------
        `ValueError`
            If the backend is not "numpy" or "cupy".
        
        Examples
        --------
        >>> # Create a 3D domain that shares the z-axis and uses cupy
        >>> domain = DomainDecomposition(
        ...     n_global=[128, 128, 128], halo=2, shared_axes=[2], backend="cupy")
        >>> # Create a 2D domain that does not share any axes and uses numpy
        >>> domain = DomainDecomposition(
        ...     n_global=[128, 128], halo=2, shared_axes=None, backend="numpy")
        """

        # set input parameters
        n_dims = len(n_global)

        # --------------------------------------------------------------
        #  Import the backend (numpy or cupy) and set the device
        # --------------------------------------------------------------
        if backend == "cupy":
            import cupy as cp
            backend_is_cupy = True
            ncp = cp
        elif backend == "numpy":
            import numpy as np
            backend_is_cupy = False
            ncp = np
        else:
            raise ValueError("Unknown backend. Use 'numpy' or 'cupy'.")

        # set the device (only important for cupy backend)
        set_device(backend_is_cupy)

        # --------------------------------------------------------------
        #  Get the number of processors in each direction
        # --------------------------------------------------------------
        # set the number of processors to 1 for shared axes
        shared_axes = shared_axes or []
        n_procs = [1 if i in shared_axes else 0 for i in range(n_dims)]
        # calculate the remaining dimensions that are not shared
        n_procs = MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), n_procs)
        # update the shared axes
        shared_axes = [i for i, n in enumerate(n_procs) if n == 1]

        # --------------------------------------------------------------
        #  Initialize the communicators
        # --------------------------------------------------------------
        comm = MPI.COMM_WORLD.Create_cart(
            n_procs, periods=[True]*n_dims, reorder=reorder_comm)
        size = comm.Get_size()
        rank = comm.Get_rank()
        

        # --------------------------------------------------------------
        #  Create subdomains
        # --------------------------------------------------------------
        all_subdomains = [Subdomain(i, comm, n_global, halo) for i in range(size)]
        my_subdomain = all_subdomains[rank]

        # check that the number of grid points in each local domain is 
        # larger than the number of halo cells
        for i in range(n_dims):
            if my_subdomain.inner_shape[i] < halo:
                raise ValueError(
                    f"Number of grid points in the direction {i} is too small. "
                    f"Add more grid points or reduce the number of halo cells.")

        # --------------------------------------------------------------
        #  Prepare the halo exchange
        # --------------------------------------------------------------
        # create subcommunicators for each axis
        subcomms = []
        for i in range(n_dims):
            subdims = [False] * n_dims
            subdims[i] = True
            subcomms.append(comm.Sub(subdims))
        
        # get the neighbors for each axis
        neighbors = [s.Shift(0, 1) for s in subcomms]
        prev_proc = [n[0] for n in neighbors]
        next_proc = [n[1] for n in neighbors]

        # create slices for halo exchange
        send_to_next = make_slice_list(slice(-2*halo, -halo), n_dims)
        send_to_prev = make_slice_list(slice(halo, 2*halo), n_dims)
        recv_from_next = make_slice_list(slice(-halo, None), n_dims)
        recv_from_prev = make_slice_list(slice(None, halo), n_dims)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        # public attributes
        self.n_dims = n_dims           # number of dimensions
        self.n_global = n_global       # total number of grid points
        self.halo = halo               # number of halo cells
        self.backend = backend         # backend (numpy or cupy)
        self.ncp = ncp                 # numpy or cupy
        self.n_procs = n_procs         # number of processors in each direction
        self.shared_axes = shared_axes # axes that are shared between processors
        self.comm = comm               # communicator
        self.size = size               # number of processors
        self.rank = rank               # rank of this processor
        self.all_subdomains = all_subdomains  # list of all subdomains
        self.my_subdomain = my_subdomain  # subdomain of this processor

        # private attributes
        self._subcomms = subcomms
        self._backend_is_cupy = backend_is_cupy
        self._next_proc = next_proc
        self._prev_proc = prev_proc
        self._send_to_next = send_to_next
        self._send_to_prev = send_to_prev
        self._recv_from_next = recv_from_next
        self._recv_from_prev = recv_from_prev
        
        
        return

    def sync(self, arr) -> None:
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
        
        Returns
        -------
        `None`
        
        Examples
        --------
        ```
        # create a domain decomposition
        domain = DomainDecomposition(n_global=[128, 128], shared_axes=[0])

        # create a random array on the local domain
        u = domain.ncp.random.rand(domain.my_subdomain.shape)

        # synchronize the halo regions between neighboring domains
        domain.sync(u)
        ```
        """
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return
        
        # synchronize cpu and gpu
        self.sync_with_device()

        # synchronize one dimension at a time
        # we need to create 4 buffers for each dimension:
        #  - buf_send_next: buffer to be sent to the next processor
        #  - buf_send_prev: buffer to be sent to the previous processor
        #  - buf_recv_next: buffer to receive data from the next processor
        #  - buf_recv_prev: buffer to receive data from the previous processor
        #
        #            buf_send_prev          buf_send_next
        # |                v                      v                |
        # |-----------|----------+-----------+---------|-----------|
        # |     ^                                            ^     |
        #  buf_recv_prev                               buf_recv_next

        for i in range(self.n_dims):
            if self.n_procs[i] == 1:
                arr[self._recv_from_next[i]] = arr[self._send_to_prev[i]]
                arr[self._recv_from_prev[i]] = arr[self._send_to_next[i]]
                continue

            reqs = []

            # sending
            buf_send_next = self.ncp.ascontiguousarray(arr[self._send_to_next[i]])
            buf_send_prev = self.ncp.ascontiguousarray(arr[self._send_to_prev[i]])
            self.sync_with_device()
            reqs.append(self._subcomms[i].Isend(
                buf_send_next, dest=self._next_proc[i], tag=0))
            reqs.append(self._subcomms[i].Isend(
                buf_send_prev, dest=self._prev_proc[i], tag=0))

            # receiving
            buf_recv_next = self.ncp.empty_like(arr[self._recv_from_next[i]])
            buf_recv_prev = self.ncp.empty_like(arr[self._recv_from_prev[i]])
            reqs.append(self._subcomms[i].Irecv(
                buf_recv_next, source=self._next_proc[i], tag=0))
            reqs.append(self._subcomms[i].Irecv(
                buf_recv_prev, source=self._prev_proc[i], tag=0))

            # wait for all non-blocking operations to complete
            MPI.Request.Waitall(reqs)

            # copy the received data to the halo regions
            arr[self._recv_from_next[i]] = buf_recv_next
            arr[self._recv_from_prev[i]] = buf_recv_prev
        return

    def sync_list(self, arr_list: 'list') -> None:
        """
        # Synchronize Multiple Arrays
        Synchronize the halo regions of a list of arrays between neighboring domains.
        
        Description
        -----------
        This function synchronizes the halo regions of a list of arrays between
        neighboring domains. The synchronization is done in place (no return value).
        Synchronization is always periodic in all directions.
        
        Parameters
        ----------
        `arr_list` : `list[ndarray]`
            List of arrays to synchronize.
        
        Returns
        -------
        `None`
        
        Examples
        --------
        ```
        # create a domain decomposition
        domain = DomainDecomposition(n_global=[128, 128], shared_axes=[0])

        # create a random array on the local domain
        u = domain.ncp.random.rand(domain.my_subdomain.shape)
        v = domain.ncp.random.rand(domain.my_subdomain.shape)

        # synchronize the halo regions between neighboring domains
        domain.sync_list([u, v])
        ```
        """
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return
        
        # synchronize cpu and gpu
        if self.backend == "cupy":
            self.ncp.cuda.Stream.null.synchronize()

        # synchronize one dimension at a time
        for i in range(self.n_dims):
            if self.n_procs[i] == 1:
                for arr in arr_list:
                    arr[self._recv_from_next[i]] = arr[self._send_to_prev[i]]
                    arr[self._recv_from_prev[i]] = arr[self._send_to_next[i]]
                continue

            reqs = []

            # sending
            for j, arr in enumerate(arr_list):
                buf_send_next = self.ncp.ascontiguousarray(arr[self._send_to_next[i]])
                buf_send_prev = self.ncp.ascontiguousarray(arr[self._send_to_prev[i]])
                reqs.append(self._subcomms[i].Isend(
                    buf_send_next, dest=self._next_proc[i], tag=j))
                reqs.append(self._subcomms[i].Isend(
                    buf_send_prev, dest=self._prev_proc[i], tag=j))

            # receiving
            buf_recv_nexts = []
            buf_recv_prevs = []
            for j, arr in enumerate(arr_list):
                buf_recv_next = self.ncp.empty_like(arr[self._recv_from_next[i]])
                buf_recv_prev = self.ncp.empty_like(arr[self._recv_from_prev[i]])
                reqs.append(self._subcomms[i].Irecv(
                    buf_recv_next, source=self._next_proc[i], tag=1))
                reqs.append(self._subcomms[i].Irecv(
                    buf_recv_prev, source=self._prev_proc[i], tag=0))
                buf_recv_nexts.append(buf_recv_next)
                buf_recv_prevs.append(buf_recv_prev)

            # wait for all non-blocking operations to complete
            MPI.Request.Waitall(reqs)

            # copy the received data to the halo regions
            for j, arr in enumerate(arr_list):
                arr[self._recv_from_next[i]] = buf_recv_nexts[j]
                arr[self._recv_from_prev[i]] = buf_recv_prevs[j]
        return


    def sync_with_device(self):
        """
        # Sync with Device
        Synchronize the gpu device with the processor.
        
        Description
        -----------
        When using the cupy backend, and calling a cupy function, the cpu does
        not wait for the gpu to finish the computation. This can lead to the cpu
        and gpu being out of sync. This function ensures that the cpu waits for
        all gpu computations to finish.
        
        Parameters
        ----------
        `None`
        
        Returns
        -------
        `None`
        """
        if self.backend == "cupy":
            self.ncp.cuda.Stream.null.synchronize()