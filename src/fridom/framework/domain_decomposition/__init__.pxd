# cython: language_level=3
# from mpi4py cimport MPI

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

    cpdef bint has_overlap(self, Subdomain other)
    cpdef tuple get_overlap_slice(self, Subdomain other)
    cpdef tuple g2l_slice(self, tuple global_slice)
    cpdef tuple l2g_slice(self, tuple local_slice)


# cdef class DomainDecomposition:
    # # public attributes
    # cdef int n_dims
    # cdef tuple n_global
    # cdef int halo
    # cdef tuple n_procs
    # cdef list shared_axes
    # cdef MPI.Cartcomm comm
    # cdef int size
    # cdef int rank
    # cdef list all_subdomains
    # cdef Subdomain my_subdomain

    # # private attributes
    # cdef list _subdomms
    # cdef list _next_proc
    # cdef list _prev_proc
    # cdef list _send_to_next
    # cdef list _send_to_prev
    # cdef list _recv_from_next
    # cdef list _recv_from_prev
    # cdef list _paddings
    # cdef list _inner

    # # methods
    # cpdef void sync(self, object arr, list flat_axes)
    # cpdef void sync_list(self, list arrs)
    # cpdef void _sync_axis(self, list arrs, int axis)
    # cpdef void _sync_axis_same_proc(self, object arrs, int axis)
    # cpdef void apply_boundary_condition(
        # self, object arr, object bc, int axis, str side)
    # cpdef void _apply_left_boundary_condition(
        # self, object arr, object bc, int axis)
    # cpdef void _apply_right_boundary_condition(
        # self, object arr, object bc, int axis)
    # cpdef void sync_with_device(self)
    # cpdef object __deepcopy__(self, dict memo)