# cython: language_level=3
from .domain_decomposition cimport DomainDecomposition

cdef class ParallelFFT:
    # public attributes
    cdef DomainDecomposition domain_in
    cdef DomainDecomposition domain_out

    # private attributes
    cdef list _forward_transforms
    cdef list _backward_transforms
    cdef list _fft_axes

    # methods
    cpdef object forward(self, object arr)
    cpdef object backward(self, object arr)
    cpdef object forward_apply(self, object arr, object apply_func)
    cpdef object backward_apply(self, object arr, object apply_func)