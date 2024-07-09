# cython: language_level=3

cdef class GridBase:
    def __init__(self, tuple N, tuple L, int n_dims):
        # attributes
        self.n_dims = n_dims        # read-only
        self.N = N
        self.L = L
        self.periodic_bounds = None
        self.inner_slice = tuple(slice(None) for _ in range(n_dims))
        self.X = None               # read-only
        self.x_global = None        # read-only
        self.x_local = None         # read-only
        self.dx = None              # read-only
        self.dV = None              # read-only

        # flags
        self.fourier_transform_available = False
        self.mpi_available = False

        # _cpu should be None
        self._cpu = None

    # ----------------------------------------------------------------
    #  Generic Methods
    # ----------------------------------------------------------------

    cpdef void setup(self, object mset):
        raise NotImplementedError

    cpdef object fft(self, object f):
        raise NotImplementedError

    cpdef object ifft(self, object f):
        raise NotImplementedError

    cpdef void sync(self, object f):
        raise NotImplementedError

    # ----------------------------------------------------------------
    #  Attribute Accessors
    # ----------------------------------------------------------------

    property n_dims:
        def __get__(self):
            return self.n_dims

    property total_grid_points:
        def __get__(self):
            return self.total_grid_points

    property periodic_bounds:
        def __get__(self):
            return self.periodic_bounds
        def __set__(self, list value):
            self.periodic_bounds = value

    property inner_slice:
        def __get__(self):
            return self.inner_slice

    property X:
        def __get__(self):
            return self.X

    property x_global:
        def __get__(self):
            return self.x_global

    property x_local:
        def __get__(self):
            return self.x_local

    property dx:
        def __get__(self):
            return self.dx

    property dV:
        def __get__(self):
            return self.dV

    # ----------------------------------------------------------------
    #  Flags Accessors
    # ----------------------------------------------------------------

    property fourier_transform_available:
        def __get__(self):
            return self.fourier_transform_available
        def __set__(self, bint value):
            self.fourier_transform_available = value

    property mpi_available:
        def __get__(self):
            return self.mpi_available
        def __set__(self, bint value):
            self.mpi_available = value

    property _cpu:
        def __get__(self):
            return self._cpu
        def __set__(self, GridBase value):
            self._cpu = value

