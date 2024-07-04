# cython: language_level=3
from mpi4py cimport MPI

cdef class Subdomain:
    cdef list n_global
    cdef int halo
    cdef int rank
    cdef list coord
    cdef list is_left_edge
    cdef list is_right_edge
    cdef tuple shape
    cdef tuple inner_shape
    cdef tuple position
    cdef tuple global_slice
    cdef tuple inner_slice

    def __init__(self, int rank, MPI.Cartcomm comm, list n_global, int halo):
        # get the processor coordinates and dimensions of the processor grid
        cdef int n_dims = len(n_global)
        cdef list coord = comm.Get_coords(rank)  # processor coordinates
        cdef list n_procs = comm.Get_topo()[0]   # number of processors in each dim.

        # check if a processor is at the edge of the global domain
        cdef list is_left_edge = [c == 0 for c in coord]
        cdef list is_right_edge = [c == n-1 for c,n in zip(coord, n_procs)]

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
        cdef list n_base = [n_grid // n_proc 
                            for n_grid, n_proc in zip(n_global, n_procs)]
        cdef list remainder = [n_grid % n_proc if (c == n_proc - 1) else 0
                               for c, n_grid, n_proc in zip(coord, n_global, n_procs)]
        cdef tuple inner_shape = tuple(n+r for n,r in zip(n_base, remainder))
        cdef tuple shape       = tuple(n + 2 * halo for n in inner_shape)
        cdef tuple inner_slice = tuple(slice(None) for _ in range(n_dims))
        if halo > 0:
            inner_slice = tuple(slice(halo, -halo) for _ in range(n_dims))

        # get the start position of the local domain in the global grid
        cdef tuple position = tuple(c * n for c,n in zip(coord, n_base))
        cdef tuple global_slice = tuple(slice(p,p+s) 
                                        for p,s in zip(position, inner_shape))

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        self.n_global      = n_global
        self.halo          = halo
        self.rank          = rank
        self.coord         = coord
        self.is_left_edge  = is_left_edge
        self.is_right_edge = is_right_edge
        self.shape         = shape
        self.inner_shape   = inner_shape
        self.position      = position
        self.global_slice  = global_slice
        self.inner_slice   = inner_slice

    # ================================================================
    #  Methods
    # ================================================================

    cpdef bint has_overlap(self, Subdomain other):
        cdef slice me, you
        for me, you in zip(self.global_slice, other.global_slice):
            if me.start >= you.stop or you.start >= me.stop:
                return False
        return True

    cpdef tuple get_overlap_slice(self, Subdomain other):
        # first get the overlap in the global coordinates
        cdef list global_overlap = []
        cdef slice me, you, tmp
        cdef int start, stop
        for me, you in zip(self.global_slice, other.global_slice):
            start = max(me.start, you.start)
            stop = min(me.stop, you.stop)
            tmp = slice(start, stop)
            global_overlap.append(tmp)

        # convert the global overlap to local coordinates
        return self.g2l_slice(tuple(global_overlap))

    cpdef tuple g2l_slice(self, tuple global_slice):
        cdef list local_slice = []
        cdef slice g, tmp
        cdef int p, start, stop
        for g, p in zip(global_slice, self.position):
            start = g.start - p + self.halo
            stop = g.stop - p + self.halo
            tmp = slice(start, stop)
            local_slice.append(tmp)
        return tuple(local_slice)

    cpdef tuple l2g_slice(self, tuple local_slice):
        cdef list global_slice = []
        cdef slice l, tmp
        cdef int p, start, stop
        for l, p, s in zip(local_slice, self.position):
            start = l.start + p - self.halo
            stop = l.stop + p - self.halo
            tmp = slice(start, stop)
            global_slice.append(tmp)
        return tuple(global_slice)

    # ================================================================
    #  Properties
    # ================================================================
    property n_global:
        def __get__(self):
            return self.n_global
    
    property halo:
        def __get__(self):
            return self.halo
    
    property rank:
        def __get__(self):
            return self.rank
    
    property coord:
        def __get__(self):
            return self.coord
    
    property is_left_edge:
        def __get__(self):
            return self.is_left_edge
    
    property is_right_edge:
        def __get__(self):
            return self.is_right_edge
    
    property shape:
        def __get__(self):
            return self.shape
    
    property inner_shape:
        def __get__(self):
            return self.inner_shape
    
    property position:
        def __get__(self):
            return self.position
    
    property global_slice:
        def __get__(self):
            return self.global_slice
    
    property inner_slice:
        def __get__(self):
            return self.inner_slice
    