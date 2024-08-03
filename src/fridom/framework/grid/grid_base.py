# Import external modules
from typing import TYPE_CHECKING
from fridom.framework import utils
# Import type information
if TYPE_CHECKING:
    from numpy import ndarray
    from fridom.framework.model_settings_base import ModelSettingsBase
    from fridom.framework.grid.transform_type import TransformType
    from fridom.framework.grid.position_base import PositionBase
    from .diff_base import DiffBase
    from .interpolation_base import InterpolationBase
    from .position_base import PositionBase

class GridBase:
    """
    Base class for all grids in the framework.
    
    Description
    -----------
    This class does not implement any functionality, but provides a template
    for all grid classes in the framework. The base class also sets default
    flags and attributes that are common to all grids. Child classes should
    override these flags and attributes as needed.

    Flags
    -----
    `fourier_transform_available` : `bool`
        Indicates whether the grid supports fast fourier transforms.
    `mpi_available` : `bool`
        Indicates whether the grid supports MPI parallelization.
    """
    _dynamic_attributes = ["_X", "_x_global", "_x_local", 
                           '_K', '_k_local', '_k_global']
    def __init__(self, n_dims: int) -> None:

        self.name = "GridBase"

        self._n_dims = n_dims
        self._N = None
        self._L = None
        self._total_grid_points = None
        self._periodic_bounds = None
        self._inner_slice = slice(None)
        self._cell_center = None
        self._X = None
        self._x_global = None
        self._x_local = None
        self._dx = None
        self._dV = None
        self._mset = None
        # spectral properties
        self._K = None
        self._k_global = None
        self._k_local = None
        self._omega_analytical = None
        self._omega_space_discrete = None
        self._omega_time_discrete = None
        # operator modules
        self._diff_mod: 'DiffBase' = None
        self._interp_mod: 'InterpolationBase' = None

        # prepare for numpy conversion (the numpy copy will be stored here)
        self._cpu = None

        # ---------------------------------------------------------------------
        #  Set default flags
        # ---------------------------------------------------------------------
        self._fourier_transform_available = False
        self._mpi_available = False

        return

    def setup(self, mset: 'ModelSettingsBase') -> None:
        """
        Initialize the grid from the model settings.
        
        Parameters
        ----------
        `mset` : `ModelSettingsBase`
            The model settings object. This is for example needed to
            determine the required halo size.
        """       
        self._diff_mod.setup(mset=mset)
        self._interp_mod.setup(mset=mset)
        return

    def fft(self, 
            arr: 'ndarray',
            transform_types: 'tuple[TransformType] | None' = None
            ) -> 'ndarray':
        """
        Perform a (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The input array.
        `transform_types` : `tuple[TransformType]` or `None` (default: `None`)
            The type of transform to apply to each non-periodic axis.
        
        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    def ifft(self, 
             arr:'ndarray',
             transform_types: 'tuple[TransformType] | None' = None
             ) -> 'ndarray':
        """
        Perform an inverse (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The input array.
        `transform_types` : `tuple[TransformType]` or `None` (default: `None`)
            The type of transform to apply to each non-periodic axis.
        
        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    def sync(self, arr: 'ndarray') -> 'ndarray':
        """
        Synchronize the halo (boundary) points of an array across all MPI ranks.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The array to synchronize.

        Returns
        -------
        `ndarray`
            The synchronized array.
        """
        raise NotImplementedError

    def sync_multi(self, arrs: 'list[ndarray]') -> 'list[ndarray]':
        """
        Synchronize the halo (boundary) points of multiple arrays across all MPI ranks.
        
        Parameters
        ----------
        `arrs` : `list[ndarray]`
            The list of arrays to synchronize.
        
        Returns
        -------
        `list[ndarray]`
            The synchronized list of arrays.
        """
        raise NotImplementedError

    def omega(self, 
              k: 'tuple[float] | tuple[ndarray]',
              use_discrete: bool = False
              ) -> 'ndarray':
        """
        Compute the dispersion relation of the model.
        
        Parameters
        ----------
        `k` : `tuple[float] | tuple[ndarray]`
            The wave numbers
        `use_discrete` : `bool` (default: False)
            Whether to include space-discretization effects.
        
        Returns
        -------
        `ndarray`
            The dispersion relation (omega(k)).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        String representation of the grid.
        """
        res = self.name
        for key, value in self.info.items():
            res += "\n  - {}: {}".format(key, value)
        return res

    # ================================================================
    #  Operators
    # ================================================================

    def diff(self, 
             arr: 'ndarray', 
             axis: int, 
             **kwargs) -> 'ndarray':
        """
        Compute the derivative of a field along an axis.

        Parameters
        ----------
        `arr` : `ndarray`
            The field to differentiate.
        `axis` : `int`
            The axis to differentiate along.

        Returns
        -------
        `ndarray`
            The derivative of the field along the specified axis. 
            (same shape as `arr`)
        """
        return self._diff_mod.diff(arr, axis, **kwargs)

    def div(self,
            arrs: 'list[ndarray]',
            axes: list[int] | None = None,
            **kwargs) -> 'ndarray':
        """
        Calculate the divergence of a vector field (\\nabla \\cdot \\vec{v}).

        Parameters
        ----------
        `arrs` : `list[ndarray]`
            The list of arrays representing the vector field.
        `axes` : `list[int]` or `None` (default: `None`)
            The axes along which to compute the divergence. If `None`, the
            divergence is computed along all axes.

        Returns
        -------
        `ndarray`
            The divergence of the vector field.
        """
        return self._diff_mod.div(arrs, axes, **kwargs)
    
    def grad(self,
             arr: 'ndarray',
             axes: list[int] | None = None,
             **kwargs) -> 'list[ndarray]':
            """
            Calculate the gradient of a scalar field (\\nabla f).
    
            Parameters
            ----------
            `arr` : `ndarray`
                The array representing the scalar field.
            `axes` : `list[int]` or `None` (default: `None`)
                The axes along which to compute the gradient. If `None`, the
                gradient is computed along all axes.
    
            Returns
            -------
            `list[ndarray]`
                The gradient of the scalar field.
            """
            return self._diff_mod.grad(arr, axes, **kwargs)

    def laplacian(self,
                  arr: 'ndarray',
                  axes: list[int] | None = None,
                  **kwargs) -> 'ndarray':
            """
            Calculate the laplacian of a scalar field (\\nabla^2 f).
    
            Parameters
            ----------
            `arr` : `ndarray`
                The array representing the scalar field.
            `axes` : `list[int]` or `None` (default: `None`)
                The axes along which to compute the laplacian. If `None`, the
                laplacian is computed along all axes.
    
            Returns
            -------
            `ndarray`
                The laplacian of the scalar field.
            """
            return self._diff_mod.laplacian(arr, axes, **kwargs)

    def curl(self,
             arrs: 'list[ndarray]',
             axes: list[int] | None = None,
             **kwargs) -> 'list[ndarray]':
            """
            Calculate the curl of a vector field (\\nabla \\times \\vec{v}).
    
            Parameters
            ----------
            `arrs` : `list[ndarray]`
                The list of arrays representing the vector field.
            `axes` : `list[int]` or `None` (default: `None`)
                The axes along which to compute the curl. If `None`, the
                curl is computed along all axes.
    
            Returns
            -------
            `list[ndarray]`
                The curl of the vector field.
            """
            return self._diff_mod.curl(arrs, axes, **kwargs)

    def interpolate(self, 
                    arr: 'ndarray', 
                    origin: 'PositionBase', 
                    destination: 'PositionBase') -> 'ndarray':
        """
        Interpolate an array from one position to another.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The array to interpolate.
        `origin` : `Position`
            The position of the array.
        `destination` : `Position`
            The position to interpolate to.
        
        Returns
        -------
        `ndarray`
            The interpolated array.
        """
        return self._interp_mod.interpolate(arr, origin, destination)

    # ----------------------------------------------------------------
    #  Grid properties
    # ----------------------------------------------------------------
    @property
    def info(self) -> dict:
        """
        Return a dictionary with information about the grid.
        
        Description
        -----------
        This method should be overridden by the child class to return a
        dictionary with information about the grid. This information is
        used to print the grid in the `__repr__` method.
        """
        return {}

    @property
    def n_dims(self) -> int:
        """The number of dimensions of the grid."""
        return self._n_dims

    @property
    def N(self) -> tuple[int]:
        """The number of grid points in each dimension."""
        return self._N

    @property
    def L(self) -> tuple[float]:
        """The length of the grid in each dimension."""
        return self._L

    @property
    def total_grid_points(self) -> int:
        """The total number of grid points in the grid."""
        return self._total_grid_points

    @property
    def periodic_bounds(self) -> list[bool]:
        """A tuple of booleans indicating whether the grid is periodic 
        in each dimension."""
        return self._periodic_bounds

    @property
    def inner_slice(self) -> tuple[slice]:
        """The slice of the grid that excludes the boundary points."""
        return self._inner_slice

    @property
    def cell_center(self) -> 'PositionBase':
        """The position of the cell centers."""
        return self._cell_center

    @cell_center.setter
    def cell_center(self, value: 'PositionBase') -> None:
        self._cell_center = value
        return

    @property
    def X(self) -> 'tuple[ndarray]':
        """The meshgrid of the grid points."""
        return self._X

    @property
    def x_global(self) -> 'tuple[ndarray]':
        """The x-vector of the global grid points."""
        return self._x_global

    @property
    def x_local(self) -> 'ndarray':
        """The x-vector of the local grid points."""
        return self._x_local

    @property
    def K(self) -> 'ndarray':
        """The wavenumber of the grid."""
        return self._K
    
    @property
    def k_global(self) -> 'ndarray':
        """The global wavenumber of the grid."""
        return self._k_global
    
    @property
    def k_local(self) -> 'ndarray':
        """The local wavenumber of the grid."""
        return self._k_local

    @property
    def dx(self) -> 'tuple[ndarray]':
        """The grid spacing in each dimension."""
        return self._dx

    @property
    def dV(self) -> 'ndarray':
        """The volume element of the grid."""
        return self._dV

    @property
    def omega_analytical(self):
        """
        Analytical dispersion relation.
        """
        if self._omega_analytical is None:
            self._omega_analytical = self.omega(self.K, use_discrete=False)
        return self._omega_analytical

    @property
    def omega_space_discrete(self):
        """
        Dispersion relation with space-discretization effects.
        """
        if self._omega_space_discrete is None:
            self._omega_space_discrete = self.omega(self.K, use_discrete=True)
        
        return self._omega_space_discrete

    @property
    def omega_time_discrete(self):
        """
        Dispersion relation with space-time-discretization effects.
        Warning: The computation may be very slow.
        """
        if self._omega_time_discrete is None:
            om_space_discrete = self.omega_space_discrete
            ts = self.mset.time_stepper
            om = ts.time_discretization_effect(om_space_discrete)
            self._omega_time_discrete = om
        return self._omega_time_discrete

    # ================================================================
    #  Flags
    # ================================================================

    @property
    def fourier_transform_available(self) -> bool:
        """Indicates whether the grid supports fast fourier transforms."""
        return self._fourier_transform_available
    @fourier_transform_available.setter
    def fourier_transform_available(self, value: bool) -> None:
        self._fourier_transform_available = value

    @property
    def mpi_available(self) -> bool:
        """Indicates whether the grid supports MPI parallelization."""
        return self._mpi_available
    @mpi_available.setter
    def mpi_available(self, value: bool) -> None:
        self._mpi_available = value

    @property
    def mset(self) -> 'ModelSettingsBase | None':
        """The model settings object."""
        return self._mset

utils.jaxify_class(GridBase)