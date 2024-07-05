# cython: language_level=3
from mpi4py cimport MPI
from mpi4py import MPI
from copy import deepcopy

from .subdomain cimport Subdomain
from fridom.framework import config

cdef list _make_slice_list(slice s, int n_dims):
    """
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
    >>> halo = 2
    >>> n_dims = 2
    >>> send_to_next = _make_slice_list(slice(-2*halo, -halo))
    >>> send_to_next
    >>> # [(slice(-4, -2), slice(None)), (slice(None), slice(-4, -2))]
    >>> # send_to_next[0] is used to send data to the next processor in the
    >>> # first dimension, send_to_next[1] is used to send data to the next
    >>> # processor in the second dimension
    """
    cdef list slice_list = []
    cdef int i
    cdef list full_slice
    for i in range(n_dims):
        full_slice = [slice(None) for _ in range(n_dims)]
        full_slice[i] = s
        slice_list.append(tuple(full_slice))
    return slice_list

cdef void set_device():
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
    cdef MPI.Comm comm_node
    cdef int node_rank, num_gpus, device_id
    if config.backend == "cupy":
        import cupy as cp
        comm_node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        node_rank = comm_node.Get_rank()
        num_gpus = cp.cuda.runtime.getDeviceCount()
        device_id = node_rank % num_gpus
        cp.cuda.Device(device_id).use()
        comm_node.Free()
    return

cdef class DomainDecomposition:
    def __init__(self, list n_global, int halo = 0, list shared_axes = None,
                 bint reorder_comm = True):
        # set input parameters
        cdef int n_dims = len(n_global)

        # set the device (only important for cupy backend)
        set_device()

        # --------------------------------------------------------------
        #  Get the number of processors in each direction
        # --------------------------------------------------------------
        # set the number of processors to 1 for shared axes
        shared_axes = shared_axes or []
        cdef list n_procs = []
        cdef int i
        for i in range(n_dims):
            if i in shared_axes:
                n_procs.append(1)
            elif n_global[i] == 1:
                n_procs.append(1)
            else:
                n_procs.append(0)
        # calculate the remaining dimensions that are not shared
        n_procs = MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), n_procs)
        # update the shared axes
        shared_axes = [i for i, n in enumerate(n_procs) if n == 1]

        # --------------------------------------------------------------
        #  Initialize the communicators
        # --------------------------------------------------------------
        cdef MPI.Cartcomm comm = MPI.COMM_WORLD.Create_cart(
            n_procs, periods=[True]*n_dims, reorder=reorder_comm)
        cdef int size = comm.Get_size()
        cdef int rank = comm.Get_rank()

        # --------------------------------------------------------------
        #  Create subdomains
        # --------------------------------------------------------------
        cdef list all_subdomains
        all_subdomains = [Subdomain(i, comm, n_global, halo) for i in range(size)]
        cdef Subdomain my_subdomain = all_subdomains[rank]

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
        cdef list paddings = [[(halo, halo) if i == j else (int(0), int(0)) 
                                for i in range(n_dims)]
                                for j in range(n_dims)]
        cdef list inner = _make_slice_list(slice(halo, -halo), n_dims)

        # create subcommunicators for each axis
        cdef list subcomms = []
        cdef list subdims
        for i in range(n_dims):
            subdims = [False] * n_dims
            subdims[i] = True
            subcomms.append(comm.Sub(subdims))
        
        # get the neighbors for each axis
        cdef list neighbors, prev_proc, next_proc
        neighbors = [s.Shift(0, 1) for s in subcomms]
        prev_proc = [n[0] for n in neighbors]
        next_proc = [n[1] for n in neighbors]

        # create slices for halo exchange
        cdef list send_to_next, send_to_prev, recv_from_next, recv_from_prev
        send_to_next = _make_slice_list(slice(-2*halo, -halo), n_dims)
        send_to_prev = _make_slice_list(slice(halo, 2*halo), n_dims)
        recv_from_next = _make_slice_list(slice(-halo, None), n_dims)
        recv_from_prev = _make_slice_list(slice(None, halo), n_dims)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        # public attributes
        self.n_dims = n_dims           # number of dimensions
        self.n_global = n_global       # total number of grid points
        self.halo = halo               # number of halo cells
        self.n_procs = n_procs         # number of processors in each direction
        self.shared_axes = shared_axes # axes that are shared between processors
        self.comm = comm               # communicator
        self.size = size               # number of processors
        self.rank = rank               # rank of this processor
        self.all_subdomains = all_subdomains  # list of all subdomains
        self.my_subdomain = my_subdomain  # subdomain of this processor

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

    cpdef void sync(self, object arr, list flat_axes = None):
        self.sync_list([arr], flat_axes)
        return

    cpdef void sync_list(self, list arrs, list flat_axes = None):
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return

        flat_axes = flat_axes or []
        
        # synchronize cpu and gpu
        self.sync_with_device()

        # synchronize one dimension at a time
        cdef int axis
        for axis in range(self.n_dims):
            if axis in flat_axes:
                continue
            self._sync_axis(arrs, axis)
        return

    cdef void _sync_axis_same_proc(self, object arrs, int axis):
        if self.n_global[axis] < self.halo:
            for arr in arrs:
                arr[:] = config.ncp.pad(
                    arr[self._inner[axis]], self._paddings[axis], mode='wrap')
        else:
            for arr in arrs:
                arr[self._recv_from_next[axis]] = arr[self._send_to_prev[axis]]
                arr[self._recv_from_prev[axis]] = arr[self._send_to_next[axis]]
        return

    cdef void _sync_axis(self, list arrs, int axis):
        if self.n_procs[axis] == 1:
            self._sync_axis_same_proc(arrs, axis)
            return

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
        cdef object ncp = config.ncp
        cdef object buf_send_next, buf_send_prev
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
        cdef list buf_recv_next_list = []
        cdef list buf_recv_prev_list = []
        cdef object buf_recv_next, buf_recv_prev
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
        MPI.Request.Waitall(reqs)

        # copy the received data to the halo regions
        cdef int i
        for i, arr in enumerate(arrs):
            arr[self._recv_from_next[axis]] = buf_recv_next_list[i]
            arr[self._recv_from_prev[axis]] = buf_recv_prev_list[i]
        return

    cpdef void sync_with_device(self):
        if config.backend == "cupy":
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    cpdef void apply_boundary_condition(
        self, object arr, object bc, int axis, str side):
        if side == "left":
            self._apply_left_boundary_condition(arr, bc, axis)
        elif side == "right":
            self._apply_right_boundary_condition(arr, bc, axis)
        else:
            raise ValueError(f"Unknown side '{side}'. Use 'left' or 'right'.")

    cdef void _apply_left_boundary_condition(
        self, object arr, object bc, int axis):
        # apply the boundary condition to the left side
        if self.my_subdomain.is_left_edge[axis]:
            arr[self._recv_from_prev[axis]] = bc
        return

    cdef void _apply_right_boundary_condition(
        self, object arr, object bc, int axis):
        # apply the boundary condition to the right side
        if self.my_subdomain.is_right_edge[axis]:
            arr[self._recv_from_next[axis]] = bc
        return

    cpdef DomainDecomposition __deepcopy__(self, dict memo):
        cdef list n_global = deepcopy(self.n_global, memo)
        cdef int halo = self.halo
        cdef list shared_axes = deepcopy(self.shared_axes, memo)
        cdef DomainDecomposition deepcopy_obj = DomainDecomposition(
            n_global, halo, shared_axes)
        memo[id(self)] = deepcopy_obj  # Store in memo to handle self-references
        return deepcopy_obj

    def __del__(self):
        self.comm.Free()
        for comm in self._subcomms:
            comm.Free()

    # ================================================================
    #  Properties
    # ================================================================
    property n_dims:
        def __get__(self):
            return self.n_dims

    property n_global:
        def __get__(self):
            return self.n_global

    property halo:
        def __get__(self):
            return self.halo

    property n_procs:
        def __get__(self):
            return self.n_procs

    property shared_axes:
        def __get__(self):
            return self.shared_axes

    property comm:
        def __get__(self):
            return self.comm
    
    property size:
        def __get__(self):
            return self.size
    
    property rank:
        def __get__(self):
            return self.rank
    
    property all_subdomains:
        def __get__(self):
            return self.all_subdomains

    property my_subdomain:
        def __get__(self):
            return self.my_subdomain