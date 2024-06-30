# Import external modules
from typing import TYPE_CHECKING
import numpy as np
# Import internal modules
from fridom.framework import config
from fridom.framework.grid.grid_base import GridBase
from fridom.framework.domain_decomposition import DomainDecomposition
from fridom.framework.domain_decomposition import ParallelFFT
from .fft import FFT
# Import type information
if TYPE_CHECKING:
    from fridom.framework.modelsettings_base import ModelSettingsBase


class CartesianGrid(GridBase):
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
    `N` : `list[int]`
        Number of grid points in each direction.
    `L` : `list[float]`
        Domain size in meters in each direction.
    `periodic_bounds` : `list[bool]`, optional
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
        Default is True for all axes.
    `shared_axes` : `list[int]`, optional
        A list of integers that indicate which axes are shared among MPI ranks.
        Default is None, which means that no fourier transforms are available.
    
    Attributes
    ----------
    `n_dims` : `int`
        The number of dimensions of the grid.
    `L` : `list[float]`
        Domain size in each direction.
    `N` : `list[int]`
        Number of grid points in each direction.
    `total_grid_points` : `int` (read-only)
        Total number of grid points.
    `dx` : `list[float]` (read-only)
        Grid spacing in each direction.
    `X` : `list[np.ndarray]` (read-only)
        Physical meshgrid on the local domain (with ghost points).
    `x_local` : `list[np.ndarray]` (read-only)
        Physical x-vectors on the local domain (without ghost points).
    `x_global` : `list[np.ndarray]` (read-only)
        Global physical x-vectors.
    `K` : `list[np.ndarray]` (read-only)
        Spectral meshgrid on the local domain.
    `k_local` : `list[np.ndarray]` (read-only)
        Spectral k-vectors on the local domain.
    `k_global` : `list[np.ndarray]` (read-only)
        Global spectral k-vectors.
    `periodic_bounds` : `list[bool]` (read-only)
        A list of booleans that indicate whether the axis is periodic.
    'subdomain_phy' : `Subdomain` (read-only)
        The subdomain of the physical domain decomposition.
    'subdomain_spe' : `Subdomain` (read-only)
        The subdomain of the spectral domain decomposition.
    `mset` : `ModelSettingsBase` (read-only)
        The model settings that the grid is constructed with.
    
    Methods
    -------
    `setup(mset: ModelSettingsBase)`
        Setup the grid (meshgrids, etc.) using the model settings.
    `fft(u: np.ndarray) -> np.ndarray`
        Forward transform from physical space to spectral space.
    `ifft(u: np.ndarray) -> np.ndarray`
        Backward transform from spectral space to physical space.
    `sync_physical(u: np.ndarray) -> None`
        Synchronize the physical field across MPI ranks.
    `sync_spectral(u: np.ndarray) -> None`
        Synchronize the spectral field across MPI ranks.
    
    Examples
    --------
    >>> import fridom.framework as fr
    >>> # construct a 3D grid:
    >>> grid = fr.grid.CartesianGrid(
    ...     N=[32, 32, 8],  # 32x32x8 grid points
    ...     L=[100.0, 100.0, 10.0],  # 100m x 100m x 10m domain
    ...     periodic_bounds=[True, True, False]  # non-periodic in z
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
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None,
                 shared_axes: list[int] | None = None) -> None:
        super().__init__()
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
        self._dx = [L / N for L, N in zip(L, N)]
        self._X: list | None = None
        self._x_local: list | None = None
        self._x_global: list | None = None
        self._K: list | None = None
        self._k_local: list | None = None
        self._k_global: list | None = None
        self._total_grid_points = int(np.prod(N))
        self._periodic_bounds = periodic_bounds
        self._shared_axes = shared_axes
        self._mset: 'ModelSettingsBase | None' = None
        self._domain_decomp: DomainDecomposition | None = None
        self._pfft: ParallelFFT | None = None
        self._fft: FFT | None = None
        return

    def setup(self, mset: 'ModelSettingsBase'):
        ncp = config.ncp
        n_dims = self.n_dims
        dtype = config.dtype_real

        # --------------------------------------------------------------
        #  Initialize the domain decomposition
        # --------------------------------------------------------------
        required_halo = 0
        for mod in mset.tendencies.module_list:
            required_halo = max(required_halo, mod.required_halo)
        domain_decomp = DomainDecomposition(
            self.N, required_halo, shared_axes=self._shared_axes)

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
        x = [ncp.linspace(0, li, ni, dtype=dtype, endpoint=False) + 0.5 * dxi
             for li, ni, dxi in zip(self._L, self._N, self._dx)]
        # get the local slice of x
        global_slice = domain_decomp.my_subdomain.global_slice
        x_local = [xi[global_slice[i]] for i, xi in enumerate(x)]
        # construct the local meshgrids (without ghost points)
        X_inner = ncp.meshgrid(*x_local, indexing='ij')
        # add ghost points
        X = [ncp.zeros(domain_decomp.my_subdomain.shape, dtype=dtype) 
             for _ in range(n_dims)]
        for i in range(n_dims):
            X[i][domain_decomp.my_subdomain.inner_slice] = X_inner[i]
            domain_decomp.sync(X[i])

        # --------------------------------------------------------------
        #  Initialize the spectral meshgrid
        # --------------------------------------------------------------
        if self.fourier_transform_available:
            spectral_subdomain = pfft.domain_out.my_subdomain
            k = fft.get_freq(self._N, self._dx)
            global_slice = spectral_subdomain.global_slice
            k_local = [ki[global_slice[i]] for i, ki in enumerate(k)]
            K = ncp.meshgrid(*k_local, indexing='ij')
        else:
            k = None
            k_local = None
            K = None

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
        return

    def fft(self, u: np.ndarray) -> np.ndarray:
        return self._pfft.forward_apply(u, self._fft.forward)

    def ifft(self, u: np.ndarray) -> np.ndarray:
        return self._pfft.backward_apply(u, self._fft.backward)

    def sync_physical(self, u: np.ndarray) -> None:
        self._domain_decomp.sync(u)

    def sync_spectral(self, u: np.ndarray) -> None:
        self._pfft.domain_out.sync(u)

    def apply_boundary_condition(self, field, axis, side, value):
        """
        Apply boundary conditions to a field.
        
        Parameters
        ----------
        `field` : `FieldVariable`
            The field to apply the boundary conditions to.
        `axis` : `int`
            The axis to apply the boundary condition to.
        `side` : `str`
            The side to apply the boundary condition to.
        `value` : `float | np.ndarray | FieldVariable`
            The value of the boundary condition.
        """
        self._domain_decomp.apply_boundary_condition(field, value, axis, side)
        return

    @property
    def L(self) -> list:
        """Domain size in each direction."""
        return self._L

    @L.setter
    def L(self, value: list):
        self._L = [float(val) for val in value]
        self._dx = [L / N for L, N in zip(self._L, self._N)]    

    @property
    def N(self) -> list:
        """Grid points in each direction."""
        return self._N
    
    @N.setter
    def N(self, value: list):
        self._N = [int(val) for val in value]
        self._dx = [L / N for L, N in zip(self._L, self._N)]
        self._total_grid_points = int(np.prod(self._N))

    @property
    def total_grid_points(self) -> int:
        """Total number of grid points."""
        return self._total_grid_points

    @property
    def dx(self) -> list:
        """Grid spacing in each direction."""
        return self._dx

    @property
    def X(self) -> list | None:
        """Physical meshgrid on the local domain."""
        return self._X

    @property
    def x_local(self) -> list | None:
        """Physical x-vectors on the local domain."""
        return self._x_local

    @property
    def x_global(self) -> list | None:
        """Global physical x-vectors."""
        return self._x_global

    @property
    def K(self) -> list | None:
        """Spectral meshgrid on the local domain."""
        return self._K
    
    @property
    def k_local(self) -> list | None:
        """Spectral k-vectors on the local domain."""
        return self._k_local
    
    @property
    def k_global(self) -> list | None:
        """Global spectral k-vectors."""
        return self._k_global

    @property
    def periodic_bounds(self) -> list:
        return self._periodic_bounds

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def subdomain_phy(self):
        return self._domain_decomp.my_subdomain

    @property
    def subdomain_spe(self):
        return self._pfft.domain_out.my_subdomain
    
    @property
    def mset(self) -> 'ModelSettingsBase':
        return self._mset