# Import external modules
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.grid.grid_base import GridBase
from fridom.framework.domain_decomposition import DomainDecomposition
from fridom.framework.domain_decomposition import ParallelFFT
from .fft import FFT
from .finite_differences import FiniteDifferences
from .linear_interpolation import LinearInterpolation
from fridom.framework.utils import humanize_number
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.transform_type import TransformType
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.domain_decomposition import Subdomain
    from fridom.framework.grid.diff_base import DiffBase
    from fridom.framework.grid.interpolation_base import InterpolationBase


class Grid(GridBase):
    """
    An n-dimensional cartesian grid with capabilities for fourier transforms.
    
    Description
    -----------
    The cartesian grid is a regular grid with constant grid spacing in each
    direction. The grid can be periodic in some directions and non-periodic in
    others. When performing a fourier transform, ffts are applied to the
    periodic axes and discrete cosine transforms to the non-periodic axes. When
    constructing the grid with MPI initialized, the grid will be distributed
    among the MPI ranks.
    
    Parameters
    ----------
    `N` : `tuple[int]`
        Number of grid points in each direction.
    `L` : `tuple[float]`
        Domain size in meters in each direction.
    `periodic_bounds` : `tuple[bool]`, optional
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
        Default is True for all axes.
    `shared_axes` : `list[int]`, optional
        A list of integers that indicate which axes are shared among MPI ranks.
        Default is None, which means that no fourier transforms are available.
    `diff_mod` : `DiffBase`, optional
        A module that contains the differentiation operators. Default is None
        which constructs the finite differences module.
    `interp_mod` : `InterpolationBase`, optional
        A module that contains the interpolation methods. Default is None
        which constructs the linear interpolation module.
    
    Attributes
    ----------
    `n_dims` : `int`
        The number of dimensions of the grid.
    `L` : `tuple[float]`
        Domain size in each direction.
    `N` : `tuple[int]`
        Number of grid points in each direction.
    `total_grid_points` : `int` (read-only)
        Total number of grid points.
    `dx` : `tuple[float]` (read-only)
        Grid spacing in each direction
    `dV` : `float` (read-only)
        Volume element.
    `X` : `tuple[np.ndarray]` (read-only)
        Physical meshgrid on the local domain (with ghost points).
    `x_local` : `tuple[np.ndarray]` (read-only)
        Physical x-vectors on the local domain (without ghost points).
    `x_global` : `tuple[np.ndarray]` (read-only)
        Global physical x-vectors.
    `K` : `tuple[np.ndarray]` (read-only)
        Spectral meshgrid on the local domain.
    `k_local` : `tuple[np.ndarray]` (read-only)
        Spectral k-vectors on the local domain.
    `k_global` : `tuple[np.ndarray]` (read-only)
        Global spectral k-vectors.
    `periodic_bounds` : `tuple[bool]` (read-only)
        A list of booleans that indicate whether the axis is periodic.
    `inner_slice` : `tuple[slice]` (read-only)
        The slice of the grid that excludes the boundary points.
    
    Methods
    -------
    `setup(mset: ModelSettingsBase)`
        Setup the grid (meshgrids, etc.) using the model settings.
    `fft(arr: np.ndarray) -> np.ndarray`
        Forward transform from physical space to spectral space.
    `ifft(arr: np.ndarray) -> np.ndarray`
        Backward transform from spectral space to physical space.
    `sync(arr: np.ndarray) -> np.ndarray`
        Synchronize the field across MPI ranks.
    `apply_boundary_condition(arr, axis, side, value) -> np.ndarray`
        Apply boundary conditions to a field.
    `get_domain_decomposition(spectral=False)`
        Get the domain decomposition of the physical or spectral domain.
    `get_subdomain(spectral=False)`
        Get the subdomain of the physical or spectral domain.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> # construct a 3D grid:
    >>> grid = fr.grid.CartesianGrid(
    ...     N=(32, 32, 8),  # 32x32x8 grid points
    ...     L=(100.0, 100.0, 10.0),  # 100m x 100m x 10m domain
    ...     periodic_bounds=(True, True, False)  # non-periodic in z
    ...     shared_axes=[0, 1]  # slab decomposition, shared in x and y
    ...     )
    >>> # setup the grid using the model settings
    >>> mset = fr.ModelSettingsBase(grid)
    >>> mset.setup()
    >>> # get the meshgrids
    >>> X, Y, Z = grid.X  # physical meshgrid of the local domain
    >>> KX, KY, KZ = grid.K  # spectral meshgrid of the local domain
    >>> # get the grid spacing
    >>> dx, dy, dz = grid.dx

    """
    _dynamic_attributes = GridBase._dynamic_attributes + ['_domain_decomp',
        '_pfft', '_fft', '_diff_mod', '_interp_mod']
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None,
                 shared_axes: list[int] | None = None,
                 diff_mod: 'DiffBase | None' = None,
                 interp_mod: 'InterpolationBase | None' = None
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
        self._domain_decomp: DomainDecomposition | None = None
        self._pfft: ParallelFFT | None = None
        self._fft: FFT | None = None
        self._diff_mod = diff_mod or FiniteDifferences()
        self._interp_mod = interp_mod or LinearInterpolation()
        return

    def setup(self, mset: 'ModelSettingsBase'):
        super().setup(mset)
        ncp = config.ncp
        n_dims = self.n_dims
        dtype = config.dtype_real

        # --------------------------------------------------------------
        #  Initialize the domain decomposition
        # --------------------------------------------------------------
        req_halo = max(self._diff_mod.required_halo, self._interp_mod.required_halo)
        req_halo = max(req_halo, mset.halo)
        domain_decomp = DomainDecomposition(
            self._N, req_halo, shared_axes=self._shared_axes)


        # --------------------------------------------------------------
        #  Initialize the fourier transform
        # --------------------------------------------------------------
        if self.fourier_transform_available:
            pfft = ParallelFFT(domain_decomp)
            fft = FFT(self._periodic_bounds)
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
            X[i] = utils.modify_array(
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
            config.logger.warning("Fourier transform not available.")
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
        return

    @partial(utils.jaxjit, static_argnames=["transform_types"])
    def fft(self, 
            arr: np.ndarray,
            transform_types: 'tuple[TransformType] | None' = None
            ) -> np.ndarray:
        f = lambda x, axes: self._fft.forward(x, axes, transform_types)
        return self._pfft.forward_apply(arr, f)

    @partial(utils.jaxjit, static_argnames=["transform_types"])
    def ifft(self, 
             arr: np.ndarray,
             transform_types: 'tuple[TransformType] | None' = None
             ) -> np.ndarray:
        f = lambda x, axes: self._fft.backward(x, axes, transform_types)
        return self._pfft.backward_apply(arr, f)

    @partial(utils.jaxjit, static_argnames=["flat_axes"])
    def sync(self, 
             arr: np.ndarray, 
             flat_axes: list[int] | None = None) -> np.ndarray:
        return self._domain_decomp.sync(arr, flat_axes=flat_axes)

    @partial(utils.jaxjit, static_argnames=["flat_axes"])
    def sync_multi(self, 
                   arrs: tuple[np.ndarray], 
                   flat_axes: list[int] | None = None) -> tuple[np.ndarray]:
        return self._domain_decomp.sync_multiple(arrs, flat_axes=flat_axes)

    @partial(utils.jaxjit, static_argnames=["axis", "side"])
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

    def get_domain_decomposition(self, spectral=False) -> DomainDecomposition:
        """
        Get the domain decomposition of the physical or spectral domain.

        Parameters
        ----------
        `spectral` : `bool`, optional
            If True, return the domain decomposition of the spectral domain.
            Default is False.
        """
        if spectral:
            return self._pfft.domain_out
        else:
            return self._domain_decomp

    def get_subdomain(self, spectral=False) -> 'Subdomain':
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
        res["L"] = humanize_number(self.L[0], "meters")
        res["dx"] = humanize_number(self.dx[0], "meters")
        res["Periodic"] = f"{self.periodic_bounds[0]}"
        for i in range(1, self.n_dims):
            res["N"] += f" x {self.N[i]}"
            res["L"] += f" x {humanize_number(self.L[i], 'meters')}"
            res["dx"] += f" x {humanize_number(self.dx[i], 'meters')}"
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

utils.jaxify_class(Grid)
