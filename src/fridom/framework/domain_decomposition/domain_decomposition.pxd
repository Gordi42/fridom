# cython: language_level=3

from .subdomain cimport Subdomain

cdef class DomainDecomposition:
    # public attributes
    cdef int n_dims
    cdef tuple n_global
    cdef int halo
    cdef tuple n_procs
    cdef list shared_axes
    cdef object comm
    cdef int size
    cdef int rank
    cdef list all_subdomains
    cdef Subdomain my_subdomain

    # private attributes
    cdef list _subdomms
    cdef list _next_proc
    cdef list _prev_proc
    cdef list _send_to_next
    cdef list _send_to_prev
    cdef list _recv_from_next
    cdef list _recv_from_prev
    cdef list _paddings
    cdef list _inner

    # methods
    cpdef void sync(self, object arr, list flat_axes=*)
    cpdef void sync_list(self, list arrs, list flat_axes=*)
    cdef void _sync_axis(self, list arrs, int axis)
    cdef void _sync_axis_same_proc(self, object arrs, int axis)
    cpdef void sync_with_device(self)
    cpdef void apply_boundary_condition(
        self, object arr, object bc, int axis, str side)
    cdef void _apply_left_boundary_condition(
        self, object arr, object bc, int axis)
    cdef void _apply_right_boundary_condition(
        self, object arr, object bc, int axis)
    cpdef object __deepcopy__(self, dict memo)