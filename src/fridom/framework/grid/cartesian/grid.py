import fridom.framework as fr
import numpy as np
from functools import partial


@fr.utils.jaxify
class Grid(fr.grid.GridBase):
    """
    An n-dimensional cartesian grid with capabilities for fourier transforms.
    
    Description
    -----------
    The cartesian grid is a regular grid with constant grid spacing in each
    direction. The grid can be periodic in some directions and non-periodic in
    others. 
    
    Parameters
    ----------
    `N` : `tuple[int]`
        Number of grid points in each direction.
    `L` : `tuple[float]`
        Domain size in meters in each direction.
    `periodic_bounds` : `tuple[bool]`, (default: None)
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
        Default is True for all axes.
    `shared_axes` : `list[int]`, (default: None)
        A list of integers that indicate which axes are shared among MPI ranks.
        Default is None, which means that no fourier transforms are available.
    `diff_mod` : `DiffModule`, (default: None)
        A module that contains the differentiation operators.
        If None, the finite differences module is used.
    `interp_mod` : `InterpolationModule`, (default: None)
        A module that contains the interpolation methods. 
        If None, the linear interpolation module is used.
    
    Examples
    --------
    .. code-block:: python

        import fridom.framework as fr
        # construct a 3D grid:
        grid = fr.grid.CartesianGrid(
            N=(32, 32, 8),  # 32x32x8 grid points
            L=(100.0, 100.0, 10.0),  # 100m x 100m x 10m domain
            periodic_bounds=(True, True, False),  # non-periodic in z
            shared_axes=[0, 1]  # slab decomposition, shared in x and y
            )
        # setup the grid using the model settings
        mset = fr.ModelSettingsBase(grid)
        mset.setup()
        # get the meshgrids
        X, Y, Z = grid.X  # physical meshgrid of the local domain
        KX, KY, KZ = grid.K  # spectral meshgrid of the local domain
        # get the grid spacing
        dx, dy, dz = grid.dx

    """
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None,
                 shared_axes: list[int] | None = None,
                 diff_mod: fr.grid.DiffModule | None = None,
                 interp_mod: fr.grid.InterpolationModule | None = None
                 ) -> None:
        super().__init__(len(N))
        self.name = "Cartesian Grid"
        # --------------------------------------------------------------
        #  Check the input
        # --------------------------------------------------------------

        # check that N and L have the same length
        if len(N) != len(L):
            raise ValueError("N and L must have the same number of dimensions.")
        n_dims = len(N)

        # check that periodic_bounds is the right length
        periodic_bounds = periodic_bounds or [True] * n_dims  # default is periodic
        if len(periodic_bounds) != n_dims:
            raise ValueError(
                "periodic_bounds must have the same number of dimensions as N and L.")

        fourier_transform_available = True
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
        # public attributes
        self._n_dims = n_dims


        # private attributes
        self._N = N
        self._L = L
        self._dx = tuple(L / N for L, N in zip(L, N))
        self._dV = np.prod(self._dx)
        self._total_grid_points = int(np.prod(N))
        self._periodic_bounds = periodic_bounds
        self._shared_axes = shared_axes
        self._domain_decomp: fr.domain_decomposition.DomainDecomposition | None = None
        self._pfft: fr.domain_decomposition.ParallelFFT | None = None
        self._fft: fr.grid.cartesian.FFT | None = None
        self._diff_module = diff_mod or fr.grid.cartesian.FiniteDifferences()
        self._interp_module = interp_mod or fr.grid.cartesian.LinearInterpolation()
        return

    def setup(self, 
              mset: 'fr.ModelSettingsBase', 
              req_halo: int | None = None,
              fft_module: 'fr.grid.cartesian.FFT | None' = None
              ) -> None:
        ncp = fr.config.ncp
        n_dims = self.n_dims
        dtype = fr.config.dtype_real

        # --------------------------------------------------------------
        #  Initialize the domain decomposition
        # --------------------------------------------------------------
        if req_halo is None:
            req_halo = max(self._diff_module.required_halo, 
                           self._interp_module.required_halo)
            req_halo = max(req_halo, mset.halo)
        domain_decomp = fr.domain_decomposition.DomainDecomposition(
            self._N, req_halo, shared_axes=self._shared_axes)


        # --------------------------------------------------------------
        #  Initialize the fourier transform
        # --------------------------------------------------------------
        if self.fourier_transform_available:
            pfft = fr.domain_decomposition.ParallelFFT(domain_decomp)
            fft = fft_module or fr.grid.cartesian.FFT(self._periodic_bounds)
        else:
            pfft = None
            fft = None

        # --------------------------------------------------------------
        #  Initialize the physical meshgrid
        # --------------------------------------------------------------
        x = tuple(ncp.linspace(0, li, ni, dtype=dtype, endpoint=False) + 0.5 * dxi
                  for li, ni, dxi in zip(self._L, self._N, self._dx))
        # get the local slice of x
        global_slice = domain_decomp.my_subdomain.global_slice
        x_local = tuple(xi[global_slice[i]] for i, xi in enumerate(x))
        # construct the local meshgrids (without ghost points)
        X_inner = ncp.meshgrid(*x_local, indexing='ij')
        # add ghost points
        X = [ncp.zeros(domain_decomp.my_subdomain.shape, dtype=dtype) 
             for _ in range(n_dims)]
        for i in range(n_dims):
            X[i] = fr.utils.modify_array(
                X[i], domain_decomp.my_subdomain.inner_slice, X_inner[i])
        X = domain_decomp.sync_multiple(tuple(X))

        # --------------------------------------------------------------
        #  Initialize the spectral meshgrid
        # --------------------------------------------------------------
        if self.fourier_transform_available:
            spectral_subdomain = pfft.domain_out.my_subdomain
            k = fft.get_freq(self._N, self._dx)
            global_slice = spectral_subdomain.global_slice
            k_local = tuple(ki[global_slice[i]] for i, ki in enumerate(k))
            K = ncp.meshgrid(*k_local, indexing='ij')
        else:
            fr.config.logger.warning("Fourier transform not available.")
            k = None
            k_local = None
            K = None

        # ----------------------------------------------------------------
        #  Store the attributes
        # ----------------------------------------------------------------

        self._mset = mset
        self._domain_decomp = domain_decomp
        self._pfft = pfft
        self._fft = fft
        self._X = X
        self._x_local = x_local
        self._x_global = x
        self._K = K
        self._k_local = k_local
        self._k_global = k
        self._inner_slice = domain_decomp.my_subdomain.inner_slice

        # call the setup method of the base class
        # This is called last since some of the setup methods of the grid base
        # class depend on the attributes set here.
        super().setup(mset)
        return

    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False
    ) -> tuple[np.ndarray]:
        if spectral:
            return self.K
        position = position or self.cell_center
        X = list(self.X)
        for i in range(self.n_dims):
            if position.positions[i] == fr.grid.AxisPosition.FACE:
                X[i] += 0.5 * self.dx[i]
        return tuple(X)

    @partial(fr.utils.jaxjit, static_argnames=["transform_types", "padding"])
    def fft(self, 
            arr: np.ndarray,
            transform_types: 'tuple[fr.grid.TransformType] | None' = None,
            padding = fr.grid.FFTPadding.NOPADDING,
            ) -> np.ndarray:
        if padding != fr.grid.FFTPadding.NOPADDING:
            raise ValueError("Padding is not supported for cartesian grids.")
        f = lambda x, axes: self._fft.forward(x, axes, transform_types)
        return self._pfft.forward_apply(arr, f)

    @partial(fr.utils.jaxjit, static_argnames=["transform_types", "padding"])
    def ifft(self, 
             arr: np.ndarray,
             transform_types: 'tuple[fr.grid.TransformType] | None' = None,
             padding = fr.grid.FFTPadding.NOPADDING,
             ) -> np.ndarray:
        if padding != fr.grid.FFTPadding.NOPADDING:
            raise ValueError("Padding is not supported for cartesian grids.")
        f = lambda x, axes: self._fft.backward(x, axes, transform_types)
        return self._pfft.backward_apply(arr, f)

    # @partial(fr.utils.jaxjit, static_argnames=["flat_axes"])
    def sync(self, 
             arr: np.ndarray, 
             flat_axes: list[int] | None = None) -> np.ndarray:
        return self._domain_decomp.sync(arr, flat_axes=flat_axes)

    @partial(fr.utils.jaxjit, static_argnames=["flat_axes"])
    def sync_multi(self, 
                   arrs: tuple[np.ndarray], 
                   flat_axes: list[int] | None = None) -> tuple[np.ndarray]:
        return self._domain_decomp.sync_multiple(arrs, flat_axes=flat_axes)

    @partial(fr.utils.jaxjit, static_argnames=["axis", "side"])
    def apply_boundary_condition(
            self, arr: 'np.ndarray', axis: int, side: str, 
            value: 'float | np.ndarray') -> np.ndarray:
        """
        Apply boundary conditions to a field.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array to apply the boundary conditions to.
        `axis` : `int`
            The axis to apply the boundary condition to.
        `side` : `str`
            The side to apply the boundary condition to.
        `value` : `float | np.ndarray`
            The value of the boundary condition.
        """
        return self._domain_decomp.apply_boundary_condition(arr, value, axis, side)

    def get_domain_decomposition(self, spectral=False
                                 ) -> fr.domain_decomposition.DomainDecomposition:
        if spectral:
            return self._pfft.domain_out
        else:
            return self._domain_decomp

    def get_subdomain(self, spectral=False) -> 'fr.domain_decomposition.Subdomain':
        """
        Get the local subdomain of the processor in the physical or spectral 
        domain decomposition.

        Parameters
        ----------
        `spectral` : `bool`, optional
            If True, return the subdomain of the spectral domain.
            Default is False.
        """
        domain_decomp = self.get_domain_decomposition(spectral)
        return domain_decomp.my_subdomain



    # ================================================================
    #  Properties
    # ================================================================
    @property
    def info(self) -> dict:
        res = super().info
        res["N"] = f"{self.N[0]}"
        res["L"] = fr.utils.humanize_number(self.L[0], "meters")
        res["dx"] = fr.utils.humanize_number(self.dx[0], "meters")
        res["Periodic"] = f"{self.periodic_bounds[0]}"
        for i in range(1, self.n_dims):
            res["N"] += f" x {self.N[i]}"
            res["L"] += f" x {fr.utils.humanize_number(self.L[i], 'meters')}"
            res["dx"] += f" x {fr.utils.humanize_number(self.dx[i], 'meters')}"
            res["Periodic"] += f" x {self.periodic_bounds[i]}"
        if self._domain_decomp is not None:
            res["Processors"] = f"{self._domain_decomp.n_procs[0]}"
            for i in range(1, self.n_dims):
                res["Processors"] += f" x {self._domain_decomp.n_procs[i]}"
        return res
        

    @property
    def L(self) -> tuple:
        """Domain size in each direction."""
        return self._L
    @L.setter
    def L(self, value: tuple):
        self._L = value
        self._dx = tuple(L / N for L, N in zip(self._L, self._N))

    @property
    def N(self) -> tuple:
        """Grid points in each direction."""
        return self._N
    @N.setter
    def N(self, value: tuple):
        self._N = value
        self._dx = tuple(L / N for L, N in zip(self._L, self._N))
        self._dV = np.prod(self._dx)
        self._total_grid_points = int(np.prod(self._N))


    @property
    def K(self) -> tuple | None:
        """Spectral meshgrid on the local domain."""
        return self._K
    
    @property
    def k_local(self) -> tuple | None:
        """Spectral k-vectors on the local domain."""
        return self._k_local
    
    @property
    def k_global(self) -> tuple | None:
        """Global spectral k-vectors."""
        return self._k_global
