# cython: language_level=3
from .domain_decomposition cimport DomainDecomposition

cdef class Transformer:
    # public attributes
    cdef DomainDecomposition domain_in
    cdef DomainDecomposition domain_out

    # private attributes
    cdef bint _same_domain
    cdef dict _overlap_info_in
    cdef dict _overlap_info_out

    # Methods
    cpdef object forward(self, arr_in, arr_out=*)
    cpdef object backward(self, arr_in, arr_out=*)