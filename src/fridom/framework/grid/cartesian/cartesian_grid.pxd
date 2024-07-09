# cython: language_level=3

# Import internal modules
from fridom.framework.grid.grid_base cimport GridBase
from fridom.framework.domain_decomposition.domain_decomposition cimport DomainDecomposition
from fridom.framework.domain_decomposition.subdomain cimport Subdomain
from fridom.framework.domain_decomposition.parallel_fft cimport ParallelFFT

cdef class CartesianGrid(GridBase):
    # public attributes
    cdef tuple K
    cdef tuple k_local
    cdef tuple k_global

    # private attributes
    cdef object _mset
    cdef DomainDecomposition _domain_decomp
    cdef ParallelFFT _pfft
    cdef object _fft
    cdef list _shared_axes

    # methods
    cpdef DomainDecomposition get_domain_decomposition(self, bint spectral=*)
    cpdef Subdomain get_subdomain(self, bint spectral=*)
    cpdef void apply_boundary_condition(
        self, object field, int axis, str side, object value)