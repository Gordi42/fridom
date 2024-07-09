# Import external modules
from typing import TYPE_CHECKING
import numpy as np
# Import internal modules
from fridom.framework import config
from fridom.framework.field_variable import FieldVariable
from fridom.framework.grid.grid_base import GridBase
from fridom.framework.domain_decomposition import DomainDecomposition
from fridom.framework.domain_decomposition import Subdomain
from fridom.framework.model_settings_base import ModelSettingsBase


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
    `fft(u: np.ndarray) -> np.ndarray`
        Forward transform from physical space to spectral space.
    `ifft(u: np.ndarray) -> np.ndarray`
        Backward transform from spectral space to physical space.
    `sync(f: FieldVariable) -> None`
        Synchronize the field across MPI ranks.
    `apply_boundary_condition(field, axis, side, value)`
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
    def __init__(self, 
                 N: tuple[int],
                 L: tuple[float],
                 periodic_bounds: tuple[bool] | None = None,
                 shared_axes: list[int] | None = None) -> None: ...

    def setup(self, mset: 'ModelSettingsBase'): ...

    def fft(self, u: np.ndarray) -> np.ndarray: ...

    def ifft(self, u: np.ndarray) -> np.ndarray: ...

    def sync(self, f: FieldVariable) -> None:
        """
        Synchronize the field across MPI ranks. (Exchange halo cells)

        Parameters
        ----------
        `f` : `FieldVariable`
            The field to synchronize.
        """

    def apply_boundary_condition(
            self, field: FieldVariable, axis: int, side: str, 
            value: float | np.ndarray | FieldVariable) -> None:
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

    def get_domain_decomposition(self, spectral: bool = False) -> DomainDecomposition:
        """
        Get the domain decomposition of the physical or spectral domain.

        Parameters
        ----------
        `spectral` : `bool`, optional
            If True, return the domain decomposition of the spectral domain.
            Default is False.
        """

    def get_subdomain(self, spectral: bool = False) -> Subdomain:
        """
        Get the local subdomain of the processor in the physical or spectral 
        domain decomposition.

        Parameters
        ----------
        `spectral` : `bool`, optional
            If True, return the subdomain of the spectral domain.
            Default is False.
        """

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def K(self) -> list | None:
        """Spectral meshgrid of the local domain."""
    
    @property
    def k_local(self) -> list | None:
        """Spectral k-vectors of the local domain."""
    
    @property
    def k_global(self) -> list | None:
        """Global spectral k-vectors."""