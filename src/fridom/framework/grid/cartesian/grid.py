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
        #  Prepare padding for FFT
        # ----------------------------------------------------------------
        # first the outer padding of trim option
        trim_zero_slice = []
        for i in range(self.n_dims):
            slices = [slice(None)] * self.n_dims
            if self._periodic_bounds[i]:
                new_kmax = int(2/3 * int(self.N[i]/2))
                slices[i] = slice(new_kmax+1, -new_kmax)
            else:
                new_kmax = int(2/3 * (self.N[i]-1))
                slices[i] = slice(new_kmax+1, None)
            trim_zero_slice.append(tuple(slices))

        # extend option
        extend_first_halfs = []
        extend_second_halfs = []
        extend_paddings = []
        extend_unpad_slices = []
        for i in range(self.n_dims):
            first_half = [slice(None)] * self.n_dims
            first_half[i] = slice(0, int((self.N[i]+1)/2))
            extend_first_halfs.append(tuple(first_half))

            second_half = [slice(None)] * self.n_dims
            second_half[i] = slice(-int(self.N[i]/2), None)
            extend_second_halfs.append(tuple(second_half))

            paddings = [(0,0)] * self.n_dims
            paddings[i] = (0, int((self.N[i]+1)/2))
            extend_paddings.append(tuple(paddings))

            sl = [slice(None)] * self.n_dims
            sl[i] = slice(0, self.N[i])
            extend_unpad_slices.append(tuple(sl))

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
        self._pad_trim_zero_slice: tuple[slice] = tuple(trim_zero_slice)
        self._extend_first_halfs: tuple[tuple[slice]] = tuple(extend_first_halfs)
        self._extend_second_halfs: tuple[tuple[slice]] = tuple(extend_second_halfs)
        self._extend_pad: tuple[tuple[int]] = tuple(extend_paddings)
        self._extend_unpad_slices: tuple[tuple[slice]] = tuple(extend_unpad_slices)

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
            ) -> np.ndarray:
        # Forward transform the array
        f = lambda x, axes: self._fft.forward(x, axes, bc_types, positions)
        u_hat = self._pfft.forward_apply(arr, f)
        
        # Apply padding if necessary
        if padding == fr.grid.FFTPadding.EXTEND:
            u_hat = self.unpad_extend(u_hat)
        return u_hat

    @partial(fr.utils.jaxjit, 
             static_argnames=["bc_types", "padding", "positions"])
    def ifft(self, 
             arr: np.ndarray,
             padding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             ) -> np.ndarray:
        # Apply padding if necessary
        match padding:
            case fr.grid.FFTPadding.NOPADDING:
                u = arr
            case fr.grid.FFTPadding.TRIM:
                u = self.pad_trim(arr)
            case fr.grid.FFTPadding.EXTEND:
                u = self.pad_extend(arr)

        f = lambda x, axes: self._fft.backward(x, axes, bc_types, positions)
        return self._pfft.backward_apply(u, f)

    def _pad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        if self._periodic_bounds[axis]:
            first_part = arr[self._extend_first_halfs[axis]]
            second_part = arr[self._extend_second_halfs[axis]]
            first_part = ncp.pad(first_part, self._extend_pad[axis], mode='constant')
            arr = ncp.concatenate((first_part, second_part), axis=axis)
        else:
            arr = ncp.pad(arr, self._extend_pad[axis], mode='constant')
        return arr

    def _unpad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        if self._periodic_bounds[axis]:
            arr = ncp.concatenate(
                (arr[self._extend_first_halfs[axis]], 
                 arr[self._extend_second_halfs[axis]]), axis=axis)
        else:
            arr = arr[self._extend_unpad_slices[axis]]
        return arr

    def pad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._pad_extend_axis(arr, axis)
        return arr

    def unpad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._unpad_extend_axis(arr, axis)
        return arr

    def pad_trim(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = fr.utils.modify_array(arr, self._pad_trim_zero_slice[axis], 0)
        return arr

    # ================================================================
    #  Syncing and Boundary Conditions
    # ================================================================

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
