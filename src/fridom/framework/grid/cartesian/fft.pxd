# cython: language_level=3

cdef class FFT:
    # public attributes
    # private attributes
    cdef tuple _periodic
    cdef list _fft_axes
    cdef list _dct_axes

    # Methods
    cpdef tuple get_freq(self, tuple shape, tuple dx)
    cpdef object forward(self, object u, list axes=*)
    cpdef object backward(self, object u_hat, list axes=*)