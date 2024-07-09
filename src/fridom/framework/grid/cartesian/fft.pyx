# cython: language_level=3
from fridom.framework import config

cdef class FFT:
    def __init__(self, list periodic):
        # --------------------------------------------------------------
        #  Check which axis to apply fft, dct
        # --------------------------------------------------------------
        cdef list fft_axes = []  # Periodic axes (fast fourier transform)
        cdef list dct_axes = []  # Non-periodic axes (discrete cosine transform)
        cdef int i
        for i in range(len(periodic)):
            if periodic[i]:
                fft_axes.append(i)
            else:
                dct_axes.append(i)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        # private attributes
        self._periodic = periodic
        self._fft_axes = fft_axes
        self._dct_axes = dct_axes
        return

    cpdef tuple get_freq(self, list shape, list dx):
        cdef object ncp = config.ncp
        cdef list k = []
        for i in range(len(shape)):
            if self._periodic[i]:
                k.append(ncp.fft.fftfreq(shape[i], dx[i]/(2*ncp.pi)))
            else:
                k.append(ncp.linspace(0, ncp.pi/dx[i], shape[i], endpoint=False))
        return tuple(k)

    cpdef object forward(self, object u, list axes = None):
        cdef object ncp = config.ncp
        cdef object scp = config.scp
        # Get the axes to apply fft, dct
        cdef list fft_axes, dct_axes
        if axes is None:
            fft_axes = self._fft_axes
            dct_axes = self._dct_axes
        else:
            fft_axes = list(set(axes) & set(self._fft_axes))
            dct_axes = list(set(axes) & set(self._dct_axes))

        # fourier transform for periodic boundary conditions
        cdef object u_hat = ncp.fft.fftn(u, axes=fft_axes)
        
        # discrete cosine transform
        for axis in dct_axes:
            u_hat = scp.fft.dct(u_hat, type=2, axis=axis)

        return u_hat


    cpdef object backward(self, object u_hat, list axes = None):
        cdef object ncp = config.ncp
        cdef object scp = config.scp
        # Get the axes to apply fft, dct
        cdef list fft_axes, dct_axes
        if axes is None:
            fft_axes = self._fft_axes
            dct_axes = self._dct_axes
        else:
            fft_axes = list(set(axes) & set(self._fft_axes))
            dct_axes = list(set(axes) & set(self._dct_axes))

        # fourier transform for periodic boundary conditions
        cdef object u = ncp.fft.ifftn(u_hat, axes=fft_axes)
        
        # discrete cosine transform
        for axis in dct_axes:
            u = scp.fft.idct(u, type=2, axis=axis)

        return u