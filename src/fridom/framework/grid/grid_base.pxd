# cython: language_level=3

cdef class GridBase:
    # attributes
    cdef int n_dims
    cdef tuple N
    cdef tuple L
    cdef int total_grid_points
    cdef tuple periodic_bounds
    cdef tuple inner_slice
    cdef tuple X
    cdef tuple x_global
    cdef tuple x_local
    cdef tuple dx
    cdef object dV

    # flags
    cdef bint fourier_transform_available
    cdef bint mpi_available

    # private attribute for deepcopy
    cdef GridBase _cpu

    # Methods
    cpdef void setup(self, object mset)
    cpdef object fft(self, object f)
    cpdef object ifft(self, object f)
    cpdef void sync(self, object f)