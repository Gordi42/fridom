# cython: language_level=3

cdef class Subdomain:
    cdef tuple n_global
    cdef int halo
    cdef int rank
    cdef tuple coord
    cdef tuple is_left_edge
    cdef tuple is_right_edge
    cdef tuple shape
    cdef tuple inner_shape
    cdef tuple position
    cdef tuple global_slice
    cdef tuple inner_slice

    cpdef bint has_overlap(self, Subdomain other)
    cpdef tuple get_overlap_slice(self, Subdomain other)
    cpdef tuple g2l_slice(self, tuple global_slice)
    cpdef tuple l2g_slice(self, tuple local_slice)