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
                 domain_decomp: fr.domain_decomposition.DomainDecomposition | None = None,
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
        periodic_bounds = tuple(periodic_bounds or [True] * n_dims)  # default is periodic
        if len(periodic_bounds) != n_dims:
            raise ValueError(
                "periodic_bounds must have the same number of dimensions as N and L.")

        fourier_transform_available = True


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
        self._domain_decomp = domain_decomp
        self._fft: fr.grid.cartesian.FFT | None = None
        self._diff_module = diff_mod or fr.grid.cartesian.FiniteDifferences()
        self._interp_module = interp_mod or fr.grid.cartesian.LinearInterpolation()
        return

    def setup(self, 
              mset: 'fr.ModelSettingsBase', 
              req_halo: int | None = None,
              fft_module: 'fr.grid.cartesian.FFT | None' = None,
              ) -> None:
        ncp = fr.config.ncp
        dtype = fr.config.dtype_real

        # --------------------------------------------------------------
        #  Initialize the domain decomposition
        # --------------------------------------------------------------
        if req_halo is None:
            req_halo = max(self._diff_module.required_halo, 
                           self._interp_module.required_halo)
            req_halo = max(req_halo, mset.halo)
        # get the domain decomposition module
        if self._domain_decomp is None:
            DomainDecomposition = fr.domain_decomposition.get_default_domain_decomposition()
            
            # construct the domain decomposition
            domain_decomp: fr.domain_decomposition.DomainDecomposition = DomainDecomposition(
                shape=tuple(self._N), 
                halo=req_halo, 
                periods=self._periodic_bounds, 
                shared_axes=None)
        else:
            domain_decomp = self._domain_decomp

        # --------------------------------------------------------------
        #  Initialize the fourier transform
        # --------------------------------------------------------------
        if self.fourier_transform_available:
            fft = fft_module or fr.grid.cartesian.FFT(self._periodic_bounds)
        else:
            fft = None

        # --------------------------------------------------------------
        #  Initialize the meshgrids
        # --------------------------------------------------------------
        x = tuple(ncp.linspace(0, li, ni, dtype=dtype, endpoint=False) + 0.5 * dxi
                  for li, ni, dxi in zip(self._L, self._N, self._dx))
        X = domain_decomp.create_meshgrid(*x, pad=True, spectral=False)

        if self.fourier_transform_available:
            k = fft.get_freq(self._N, self._dx)
            K = domain_decomp.create_meshgrid(*k, pad=False, spectral=True)
        else:
            fr.log.warning("Fourier transform not available.")
            k = None
            K = None


        # ----------------------------------------------------------------
        #  Store the attributes
        # ----------------------------------------------------------------

        self._mset = mset
        self._domain_decomp = domain_decomp
        self._fft = fft
        self._X = X
        self._x_global = x
        self._K = K
        self._k_global = k

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

    # ================================================================
    #  Fourier Transforms
    # ================================================================
    @partial(fr.utils.jaxjit, 
             static_argnames=["bc_types", "padding", "positions"])
    def fft(self, 
            arr: np.ndarray,
            padding = fr.grid.FFTPadding.NOPADDING,
            bc_types: tuple[fr.grid.BCType] | None = None,
            positions: tuple[fr.grid.AxisPosition] | None = None,
            axes: tuple[int] | None = None,
            ) -> np.ndarray:
        # Forward transform the array
        f = lambda x, axes: self._fft.forward(x, axes, bc_types, positions)
        forward = self._domain_decomp.parallel_forward_transform(f)
        u_hat = forward(arr, axes)
        
        # Apply padding if necessary
        if padding == fr.grid.FFTPadding.EXTEND:
            u_hat = self.domain_decomp.unpad_extend(u_hat)
        return u_hat

    @partial(fr.utils.jaxjit, 
             static_argnames=["bc_types", "padding", "positions"])
    def ifft(self, 
             arr: np.ndarray,
             padding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             axes: tuple[int] | None = None,
             ) -> np.ndarray:
        # Apply padding if necessary
        match padding:
            case fr.grid.FFTPadding.NOPADDING:
                u = arr
            case fr.grid.FFTPadding.TRIM:
                u = self.domain_decomp.pad_trim(arr)
            case fr.grid.FFTPadding.EXTEND:
                u = self.domain_decomp.pad_extend(arr)

        f = lambda x, axes: self._fft.backward(x, axes, bc_types, positions)
        backward = self._domain_decomp.parallel_backward_transform(f)
        return backward(u, axes)

    # ================================================================
    #  Syncing and Boundary Conditions
    # ================================================================

    @fr.utils.jaxjit
    def sync_multi(self, 
                   arrs: tuple[np.ndarray]) -> tuple[np.ndarray]:
        return self.domain_decomp.sync_multiple(arrs)

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
