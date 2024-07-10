# cython: language_level=3

# Import external modules
import numpy as np
# Import internal modules
from fridom.framework import config
from fridom.framework.to_numpy import to_numpy
from fridom.framework.grid.grid_base cimport GridBase
from fridom.framework.domain_decomposition.domain_decomposition cimport DomainDecomposition
from fridom.framework.domain_decomposition.parallel_fft cimport ParallelFFT
from fridom.framework.domain_decomposition.subdomain cimport Subdomain
from .fft import FFT


cdef class CartesianGrid(GridBase):
    def __init__(self, 
                 tuple N, 
                 tuple L, 
                 tuple periodic_bounds = None, 
                 list shared_axes = None):
        super().__init__(N, L, len(N))
        # --------------------------------------------------------------
        #  Check the input
        # --------------------------------------------------------------

        # check that N and L have the same length
        if len(N) != len(L):
            raise ValueError("N and L must have the same number of dimensions.")

        # check that periodic_bounds is the right length
        if periodic_bounds is None:
            periodic_bounds = tuple([True] * self.n_dims)
        if len(periodic_bounds) != self.n_dims:
            raise ValueError(
                "periodic_bounds must have the same number of dimensions as N and L.")

        cdef bint fourier_transform_available = True
        if shared_axes is None:
            fourier_transform_available = False


        # --------------------------------------------------------------
        #  Set the flags
        # --------------------------------------------------------------
        self.fourier_transform_available = fourier_transform_available
        self.mpi_available = True

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------
        self.dx = tuple(L / N for L, N in zip(L, N))
        self.dV = np.prod(self.dx)
        self.total_grid_points = int(np.prod(N))
        self.periodic_bounds = periodic_bounds
        # private attributes
        self._shared_axes = shared_axes
        return

    cpdef void setup(self, object mset):
        cdef object ncp = config.ncp
        cdef int n_dims = self.n_dims
        cdef object dtype = config.dtype_real

        # --------------------------------------------------------------
        #  Initialize the domain decomposition
        # --------------------------------------------------------------
        cdef int required_halo = mset.tendencies.required_halo
        cdef DomainDecomposition domain_decomp = DomainDecomposition(
            self.N, required_halo, shared_axes=self._shared_axes)

        # --------------------------------------------------------------
        #  Initialize the fourier transform
        # --------------------------------------------------------------
        cdef ParallelFFT pfft
        cdef object fft
        if self.fourier_transform_available:
            pfft = ParallelFFT(domain_decomp)
            fft = FFT(self.periodic_bounds)
        else:
            pfft = None
            fft = None

        # --------------------------------------------------------------
        #  Initialize the physical meshgrid
        # --------------------------------------------------------------
        cdef tuple x
        x = tuple([ncp.linspace(0, li, ni, dtype=dtype, endpoint=False) + 0.5 * dxi
             for li, ni, dxi in zip(self.L, self.N, self.dx)])
        # get the local slice of x
        cdef tuple global_slice = domain_decomp.my_subdomain.global_slice
        cdef tuple x_local = tuple([xi[sl] for xi, sl in zip(x, global_slice)])
        # construct the local meshgrids (without ghost points)
        cdef object X_inner = tuple(ncp.meshgrid(*x_local, indexing='ij'))
        # add ghost points
        cdef tuple X = tuple([ncp.zeros(domain_decomp.my_subdomain.shape, dtype=dtype) 
             for _ in range(n_dims)])
        for i in range(n_dims):
            X[i][domain_decomp.my_subdomain.inner_slice] = X_inner[i]
            domain_decomp.sync(X[i])

        # # --------------------------------------------------------------
        # #  Initialize the spectral meshgrid
        # # --------------------------------------------------------------
        cdef Subdomain spectral_subdomain
        cdef tuple k, k_local, K
        if self.fourier_transform_available:
            spectral_subdomain = pfft.domain_out.my_subdomain
            k = fft.get_freq(self.N, self.dx)
            global_slice = spectral_subdomain.global_slice
            k_local = tuple([ki[sl] for ki, sl in zip(k, global_slice)])
            K = tuple(ncp.meshgrid(*k_local, indexing='ij'))
        else:
            k = None
            k_local = None
            K = None

        self._mset = mset
        self._domain_decomp = domain_decomp
        self._pfft = pfft
        self._fft = fft
        self.X = X
        self.x_local = x_local
        self.x_global = x
        self.K = K
        self.k_local = k_local
        self.k_global = k
        return

    cpdef object fft(self, object u):
        return self._pfft.forward_apply(u, self._fft.forward)

    cpdef object ifft(self, object u):
        return self._pfft.backward_apply(u, self._fft.backward)

    cpdef void sync(self, object f):
        if f.is_spectral:
            self._pfft.domain_out.sync(f)
        else:
            self._domain_decomp.sync(f)

    cpdef void apply_boundary_condition(self, 
                                        object field, 
                                        int axis, 
                                        str side, 
                                        object value):
        self._domain_decomp.apply_boundary_condition(field, value, axis, side)
        return

    cpdef DomainDecomposition get_domain_decomposition(self, bint spectral = False):
        if spectral:
            return self._pfft.domain_out
        else:
            return self._domain_decomp

    cpdef Subdomain get_subdomain(self, bint spectral = False):
        return self.get_domain_decomposition(spectral).my_subdomain

    # def _to_numpy(self, memo):
        # from copy import deepcopy
        # from fridom.framework.to_numpy import to_numpy
        # cdef CartesianGrid copy = deepcopy(self)
        # for attr in dir(copy):
            # if attr.startswith("__"):
                # continue
            # if callable(getattr(copy, attr)):
                # continue
            # setattr(copy, attr, to_numpy(getattr(copy, attr), memo))
        # return copy

    def _to_numpy(self, memo):
        if self._cpu is not None:
            return self._cpu
        new_grid = CartesianGrid(self.N, self.L, self.periodic_bounds)
        new_grid.X = to_numpy(self.X, memo)
        new_grid.x_local = to_numpy(self.x_local, memo)
        new_grid.x_global = to_numpy(self.x_global, memo)
        new_grid.dx = to_numpy(self.dx, memo)
        new_grid.dV = to_numpy(self.dV, memo)
        new_grid.K = to_numpy(self.K, memo)
        new_grid.k_local = to_numpy(self.k_local, memo)
        new_grid.k_global = to_numpy(self.k_global, memo)
        self._cpu = new_grid
        return new_grid

    # ================================================================
    #  Properties
    # ================================================================
    property N:
        def __get__(self):
            return self.N
        def __set__(self, tuple value):
            self.N = value
            self.dx = tuple(L / N for L, N in zip(self.L, self.N))
            self.dV = np.prod(self.dx)
            self.total_grid_points = int(np.prod(self.N))
            return

    property L:
        def __get__(self):
            return self.L
        def __set__(self, tuple value):
            self.L = value
            self.dx = tuple(L / N for L, N in zip(self.L, self.N))
            self.dV = np.prod(self.dx)
            return

    property K:
        def __get__(self):
            return self.K

    property k_local:
        def __get__(self):
            return self.k_local

    property k_global:
        def __get__(self):
            return self.k_global

    property inner_slice:
        def __get__(self):
            return self.get_subdomain().inner_slice