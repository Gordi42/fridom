# Import external modules
from copy import deepcopy
import numpy as np
from numpy import ndarray
from functools import partial
# Import internal modules
from .subdomain import Subdomain
from fridom.framework import config, utils

def _make_slice_tuple(s: slice, n_dims: int):
    """
    Create a tuple of slices for halo exchange.
        
    Description
    -----------
    Lets assume we want to access the halo region that is shared with the
    previous processor. For a 2D domain, two such regions exist, one in the
    x-direction and one in the y-direction. Hence we need to create two
    slices, one for each dimension. In this example and a halo of 2, the
    slices would be (slice(0, 2), slice(None)) and 
    (slice(None), slice(0, 2)). This function for s=slice(0, 2) would
    return the tuple of the two slice-tuples.
        
    Parameters
    ----------
    `s` : `slice`
        A 1D slice that defines the halo region in one dimension.
    `n_dims` : `int`
        The number of dimensions.
        
    Returns
    -------
    `tuple[tuple[slice]]`
        A list of slice tuples for each dimension.
        
    Examples
    --------

    .. code-block:: python

        halo = 2
        n_dims = 2
        send_to_next = _make_slice_tuple(slice(-2*halo, -halo))
        send_to_next
        # ((slice(-4, -2), slice(None)), (slice(None), slice(-4, -2)))
        # send_to_next[0] is used to send data to the next processor in the
        # first dimension, send_to_next[1] is used to send data to the next
        # processor in the second dimension
    """
    slice_list = []
    for i in range(n_dims):
        full_slice = [slice(None)]*n_dims
        full_slice[i] = s
        slice_list.append(tuple(full_slice))
    return tuple(slice_list)

def set_device():
    """
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
    """
    if not utils.MPI_AVAILABLE:
        return

    if config.backend == "cupy":
        cp = config.ncp
        comm_node = utils.MPI.COMM_WORLD.Split_type(utils.MPI.COMM_TYPE_SHARED)
        node_rank = comm_node.Get_rank()
        num_gpus = cp.cuda.runtime.getDeviceCount()
        device_id = node_rank % num_gpus
        cp.cuda.Device(device_id).use()
        comm_node.Free()
    return



@utils.jaxify
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
    `n_global` : `tuple[int]`
        The total number of grid points in each dimension.
    `halo` : `int`, optional (default=0)
        The number of halo cells (ghost cells) around the local domain
        for the exchange of boundary values.
    `shared_axes` : `list[int]`, optional (default=None)
        A list of axes that are shared between processors.
    `reorder_comm` : `bool`, optional (default=True)
        Whether to reorder the communicator.
    
    Examples
    --------
    .. code-block:: python

        from fridom.framework import config
        from fridom.framework.domain_decomposition import DomainDecomposition
        # create a domain decomposition that shares the x-axis
        dom = DomainDecomposition(
            n_global=(128, 128), halo=2, shared_axes=[0])
        
        # create a random array on the local domain
        u = config.ncp.random.rand(*dom.my_subdomain.shape)
        
        # synchronize the halo regions between neighboring domains
        dom.sync(u)
    """
    def __init__(self, 
                 n_global: 'tuple[int]', 
                 halo: int = 0,
                 shared_axes: 'list[int] | None' = None,
                 reorder_comm = True,
                 ) -> None:
        # set input parameters
        n_dims = len(n_global)

        # set the device (only important for cupy backend)
        set_device()

        # --------------------------------------------------------------
        #  Get the number of processors in each direction
        # --------------------------------------------------------------
        if utils.MPI_AVAILABLE:
            n_procs_total = utils.MPI.COMM_WORLD.Get_size()
        else:
            n_procs_total = 1
        # set the number of processors to 1 for shared axes
        shared_axes = shared_axes or []
        n_procs = []
        for i in range(n_dims):
            if i in shared_axes:
                n_procs.append(1)
            elif n_global[i] == 1:
                n_procs.append(1)
            else:
                n_procs.append(0)
        # calculate the remaining dimensions that are not shared
        if utils.MPI_AVAILABLE:
            n_procs = utils.MPI.Compute_dims(utils.MPI.COMM_WORLD.Get_size(), n_procs)
        else:
            n_procs = (1,)*n_dims
        # update the shared axes
        shared_axes = [i for i, n in enumerate(n_procs) if n == 1]

        # --------------------------------------------------------------
        #  Initialize the communicators
        # --------------------------------------------------------------
        if utils.MPI_AVAILABLE:
            comm = utils.MPI.COMM_WORLD.Create_cart(
                n_procs, periods=[True]*n_dims, reorder=reorder_comm)
            size = comm.Get_size()
            rank = comm.Get_rank()
        else:
            comm = None
            size = 1
            rank = 0

        # --------------------------------------------------------------
        #  Create subdomains
        # --------------------------------------------------------------
        all_subdomains = [Subdomain(i, comm, n_global, halo) for i in range(size)]
        my_subdomain = all_subdomains[rank]

        # check that the number of grid points in each local domain is 
        # larger than the number of halo cells
        for i in range(n_dims):
            if n_procs[i] == 1:  # skip shared axes
                continue
            if my_subdomain.inner_shape[i] < halo:
                raise ValueError(
                    f"Number of grid points in the direction {i} is too small. "
                    f"Add more grid points or reduce the number of halo cells.")

        # --------------------------------------------------------------
        #  Prepare the halo exchange
        # --------------------------------------------------------------
        # exchange of halo regions in shared axes:
        paddings = tuple(tuple((halo, halo) if i == j else (0, 0) 
                               for i in range(n_dims))
                         for j in range(n_dims))
        inner = _make_slice_tuple(slice(halo, -halo), n_dims)

        # create subcommunicators for each axis
        if utils.MPI_AVAILABLE:
            subcomms = []
            for i in range(n_dims):
                subdims = [False] * n_dims
                subdims[i] = True
                subcomms.append(comm.Sub(subdims))
        
            # get the neighbors for each axis
            neighbors = [s.Shift(0, 1) for s in subcomms]
            prev_proc = [n[0] for n in neighbors]
            next_proc = [n[1] for n in neighbors]
        else:
            subcomms = []
            prev_proc = [0]*n_dims
            next_proc = [0]*n_dims

        # create slices for halo exchange
        send_to_next = _make_slice_tuple(slice(-2*halo, -halo), n_dims)
        send_to_prev = _make_slice_tuple(slice(halo, 2*halo), n_dims)
        recv_from_next = _make_slice_tuple(slice(-halo, None), n_dims)
        recv_from_prev = _make_slice_tuple(slice(None, halo), n_dims)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        # public readable attributes
        self._n_dims = n_dims           # number of dimensions
        self._n_global = n_global       # total number of grid points
        self._halo = halo               # number of halo cells
        self._n_procs_total = n_procs_total  # total number of processors
        self._n_procs = n_procs         # number of processors in each direction
        self._shared_axes = shared_axes # axes that are shared between processors
        self._comm = comm               # communicator
        self._size = size               # number of processors
        self._rank = rank               # rank of this processor
        self._all_subdomains = all_subdomains  # list of all subdomains
        self._my_subdomain = my_subdomain  # subdomain of this processor

        # private attributes
        self._subcomms = subcomms
        self._next_proc = next_proc
        self._prev_proc = prev_proc
        self._send_to_next = send_to_next
        self._send_to_prev = send_to_prev
        self._recv_from_next = recv_from_next
        self._recv_from_prev = recv_from_prev
        self._paddings = paddings
        self._inner = inner
        return

    def sum(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Computes the global sum of an array that is distributed over the processors.       

        Parameters
        ----------
        `arr` : `ndarray`
            The array to sum over.
        `axes` : `list[int]`, optional (default=None)
            The axes to sum over. If None, all axes are summed over.
        
        Returns
        -------
        `ndarray`
            The global sum of the array. Dimensions that are not summed over
            are kept.
        """
        if axes is not None:
            raise NotImplementedError("The `axes` argument is not yet implemented.")
        # exclude the halo regions and sum over the local domain
        ics = self.my_subdomain.inner_slice
        local_sum = arr[ics].sum()
        # sum over all processors
        if utils.MPI_AVAILABLE:
            glob_sum = utils.MPI.COMM_WORLD.allreduce(local_sum, op=utils.MPI.SUM)
        else:
            glob_sum = local_sum
        return glob_sum

    def max(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Computes the global maximum of an array that is distributed over the processors.       

        Parameters
        ----------
        `arr` : `ndarray`
            The array to compute the maximum of.
        `axes` : `list[int]`, optional (default=None)
            The axes to compute the maximum over. If None, all axes are used.
        
        Returns
        -------
        `ndarray`
            The global maximum of the array. Dimensions that are not used
            are kept.
        """
        if axes is not None:
            raise NotImplementedError("The `axes` argument is not yet implemented.")
        # exclude the halo regions and compute the maximum over the local domain
        ics = self.my_subdomain.inner_slice
        local_max = arr[ics].max()
        # compute the maximum over all processors
        if utils.MPI_AVAILABLE:
            glob_max = utils.MPI.COMM_WORLD.allreduce(local_max, op=utils.MPI.MAX)
        else:
            glob_max = local_max
        return glob_max

    def min(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        """
        Computes the global minimum of an array that is distributed over the processors.       

        Parameters
        ----------
        `arr` : `ndarray`
            The array to compute the minimum of.
        `axes` : `list[int]`, optional (default=None)
            The axes to compute the minimum over. If None, all axes are used.
        
        Returns
        -------
        `ndarray`
            The global minimum of the array. Dimensions that are not used
            are kept.
        """
        if axes is not None:
            raise NotImplementedError("The `axes` argument is not yet implemented.")
        # exclude the halo regions and compute the minimum over the local domain
        ics = self.my_subdomain.inner_slice
        local_min = arr[ics].min()
        # compute the minimum over all processors
        if utils.MPI_AVAILABLE:
            glob_min = utils.MPI.COMM_WORLD.allreduce(local_min, op=utils.MPI.MIN)
        else:
            glob_min = local_min
        return glob_min

    @partial(utils.jaxjit, static_argnames='flat_axes')
    def sync(self, arr: ndarray, flat_axes: list | None = None) -> ndarray:
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

        Returns
        -------
        `ndarray`
            The array with the halo regions synchronized.
        
        Examples
        --------
        .. code-block:: python

            from fridom.framework import config
            from fridom.framework.domain_decomposition import DomainDecomposition
            # create a domain decomposition
            domain = DomainDecomposition(n_global=(128, 128), shared_axes=[0])
            # create a random array on the local domain
            u = config.ncp.random.rand(domain.my_subdomain.shape)
            # synchronize the halo regions between neighboring domains
            u = domain.sync(u)
        """
        return self.sync_multiple((arr,), flat_axes)[0]

    @partial(utils.jaxjit, static_argnames='flat_axes')
    def sync_multiple(self, 
                      arrs: 'tuple[np.ndarray]', 
                      flat_axes: list | None = None) -> tuple[np.ndarray]:
        """
        Synchronize the halo regions of multiple arrays between neighboring domains.
        
        Description
        -----------
        This function synchronizes the halo regions of a tuple of arrays between
        neighboring domains. The synchronization is done in place (no return value).
        Synchronization is always periodic in all directions.
        
        Parameters
        ----------
        `arrs` : `tuple[ndarray]`
            List of arrays to synchronize.
        `flat_axes` : `list[int]`, optional (default=None)
            A list of axes to skip synchronization.
        
        Returns
        -------
        `tuple[ndarray]`
            The tuple of arrays with the halo regions synchronized.
        
        Examples
        --------
        .. code-block:: python

            from fridom.framework import config
            from fridom.framework.domain_decomposition import DomainDecomposition
            # create a domain decomposition
            domain = DomainDecomposition(n_global=(128, 128), shared_axes=[0])
            # create a random array on the local domain
            u = config.ncp.random.rand(domain.my_subdomain.shape)
            v = config.ncp.random.rand(domain.my_subdomain.shape)
            # synchronize the halo regions between neighboring domains
            u, v = domain.sync_multiple((u, v))
        """
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return arrs

        flat_axes = flat_axes or []
        
        # synchronize cpu and gpu
        self.sync_with_device()

        # synchronize one dimension at a time
        for axis in range(self.n_dims):
            if axis in flat_axes:
                continue
            if self._n_procs[axis] == 1:
                arrs = self._sync_axis_same_proc(arrs, axis)
            else:
                arrs = self._sync_axis(arrs, axis)
        return arrs

    @partial(utils.jaxjit, static_argnames=['axis'])
    def _sync_axis_same_proc(self, arrs: tuple[np.ndarray], 
                            axis: int,) -> tuple[np.ndarray]:
        if self.n_global[axis] < self.halo:
            pad = config.ncp.pad
            ics = self._inner[axis]
            pad_width = self._paddings[axis]
            return tuple(pad(arr[ics], pad_width, mode='wrap') for arr in arrs)
        else:
            rfn = self._recv_from_next[axis]
            rfp = self._recv_from_prev[axis]
            stn = self._send_to_next[axis]
            stp = self._send_to_prev[axis]
            if config.backend_is_jax:
                arrs = tuple(arr.at[rfn].set(arr[stp]) for arr in arrs)
                arrs = tuple(arr.at[rfp].set(arr[stn]) for arr in arrs)
            else:
                for arr in arrs:
                    arr[rfn] = arr[stp]
                    arr[rfp] = arr[stn]
            return arrs

    @partial(utils.jaxjit, static_argnames='axis')
    def _sync_axis(self, arrs: tuple[np.ndarray], axis: int) -> tuple[np.ndarray]:
        reqs = []

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

        # ----------------------------------------------------------------
        #  Sending
        # ----------------------------------------------------------------
        ncp = config.ncp
        for arr in arrs:
            buf_send_next = ncp.ascontiguousarray(arr[self._send_to_next[axis]])
            buf_send_prev = ncp.ascontiguousarray(arr[self._send_to_prev[axis]])
            self.sync_with_device()
            reqs.append(self._subcomms[axis].Isend(
                buf_send_next, dest=self._next_proc[axis], tag=0))
            reqs.append(self._subcomms[axis].Isend(
                buf_send_prev, dest=self._prev_proc[axis], tag=0))

        # ----------------------------------------------------------------
        #  Receiving
        # ----------------------------------------------------------------
        buf_recv_next_list = []
        buf_recv_prev_list = []
        for arr in arrs:
            buf_recv_next = ncp.empty_like(arr[self._recv_from_next[axis]])
            buf_recv_prev = ncp.empty_like(arr[self._recv_from_prev[axis]])
            reqs.append(self._subcomms[axis].Irecv(
                buf_recv_next, source=self._next_proc[axis], tag=0))
            reqs.append(self._subcomms[axis].Irecv(
                buf_recv_prev, source=self._prev_proc[axis], tag=0))
            buf_recv_next_list.append(buf_recv_next)
            buf_recv_prev_list.append(buf_recv_prev)

        # wait for all non-blocking operations to complete
        utils.MPI.Request.Waitall(reqs)

        # copy the received data to the halo regions
        for i, arr in enumerate(arrs):
            arr[self._recv_from_next[axis]] = buf_recv_next_list[i]
            arr[self._recv_from_prev[axis]] = buf_recv_prev_list[i]
        return arrs

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
        if config.backend == "cupy":
            config.ncp.cuda.Stream.null.synchronize()

    def __deepcopy__(self, memo) -> 'DomainDecomposition':
        # it should never be necessary to deepcopy the domain decomposition
        # communicator, so do a shallow copy instead
        from copy import copy
        copied = copy(self)
        memo[id(self)] = copied
        return copy(self)

        # this was the original implementation, but it's probably not necessary
        deepcopy_obj = object.__new__(self.__class__)
        memo[id(self)] = deepcopy_obj  # Store in memo to handle self-references
        for key, value in vars(self).items():
            if isinstance(value, list):
                list_copy = []
                for item in value:
                    if isinstance(item, utils.MPI.Cartcomm):
                        list_copy.append(item)
                    else:
                        list_copy.append(deepcopy(item, memo))
                setattr(deepcopy_obj, key, list_copy)
            # check if value is of type mpi4py.MPI.Cartcomm
            elif isinstance(value, MPI.Cartcomm):
                setattr(deepcopy_obj, key, value)
            else:
                setattr(deepcopy_obj, key, deepcopy(value, memo))
        return deepcopy_obj

    def __del__(self):
        try:
            self._comm.Free()
        except:
            pass
        return

    def __reduce__(self):
        args = (self._n_global, self._halo, self._shared_axes)
        return (self.__class__, args)

    def __setstate__(self, state):
        self.__init__(*state)
        return self

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def n_dims(self) -> int:
        """The number of dimensions."""
        return self._n_dims

    @property
    def n_global(self) -> tuple[int]:
        """The total number of grid points in each dimension."""
        return self._n_global

    @property
    def halo(self) -> int:
        """The number of halo cells (ghost cells) around the local domain."""
        return self._halo

    @property
    def n_procs_total(self) -> int:
        """The total number of processors."""
        return self._n_procs_total

    @property
    def n_procs(self) -> tuple[int]:
        """The number of processors in each direction."""
        return self._n_procs

    @property
    def shared_axes(self) -> list[int]:
        """A list of axes that are shared between processors."""
        return self._shared_axes

    @property
    def comm(self):
        """The cartesian communicator that defines the processor grid."""
        return self._comm

    @property
    def size(self) -> int:
        """The total number of processors."""
        return self._size

    @property
    def rank(self) -> int:
        """The rank of the current processor."""
        return self._rank

    @property
    def all_subdomains(self) -> list[Subdomain]:
        """A list of all subdomains in the global domain."""
        return self._all_subdomains

    @property
    def my_subdomain(self) -> Subdomain:
        """The local domain of the current processor."""
        return self._my_subdomain
