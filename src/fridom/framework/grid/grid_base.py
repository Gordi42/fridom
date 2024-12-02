import fridom.framework as fr
from numpy import ndarray
from abc import abstractmethod
from functools import partial


@partial(fr.utils.jaxify, dynamic=('_X', '_x_global', '_K', '_k_global'))
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
    def __init__(self, n_dims: int) -> None:

        self.name = "GridBase"

        self._n_dims = n_dims
        self._N = None
        self._L = None
        self._total_grid_points = None
        self._periodic_bounds = None
        self._X = None
        self._x_global = None
        self._x_local = None
        self._dx = None
        self._dV = None
        self._mset = None
        self._water_mask = fr.grid.WaterMask()
        # The domain decomposition
        self._domain_decomposition = None
        # The cell center
        CENTER = fr.grid.AxisPosition.CENTER
        self._cell_center = fr.grid.Position(tuple([CENTER] * n_dims))
        # spectral properties
        self._K = None
        self._k_global = None
        self._k_local = None
        self._omega_analytical = None
        self._omega_space_discrete = None
        self._omega_time_discrete = None
        # operator modules
        self._diff_module: fr.grid.DiffModule = None
        self._interp_module: fr.grid.InterpolationModule = None

        # prepare for numpy conversion (the numpy copy will be stored here)
        self._cpu = None

        # ---------------------------------------------------------------------
        #  Set default flags
        # ---------------------------------------------------------------------
        self._fourier_transform_available = False
        self._mpi_available = False
        self._spectral_grid = False

        return

    def setup(self, mset: fr.ModelSettingsBase) -> None:
        """
        Initialize the grid from the model settings.
        
        Parameters
        ----------
        `mset` : `ModelSettingsBase`
            The model settings object. This is for example needed to
            determine the required halo size.
        """       
        self._diff_module.setup(mset=mset)
        self._interp_module.setup(mset=mset)
        return

    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False
    ) -> tuple[ndarray]:
        """
        Get the meshgrid of the grid points.
        
        Parameters
        ----------
        `position` : `Position` or `None` (default: `None`)
            The position of the field.
        `spectral` : `bool` (default: `False`)
            Whether to return the meshgrid of the spectral domain.
        
        Returns
        -------
        `tuple[ndarray]`
            The meshgrid of the grid points.
        """
        if position is None:
            position = self.cell_center
        if position != self.cell_center:
            raise NotImplementedError("Not implemented for this grid")
        if spectral:
            return self._K
        return self._X

    # ----------------------------------------------------------------
    #  Fourier Transform Methods
    # ----------------------------------------------------------------

    @abstractmethod
    def fft(self, 
             arr: ndarray,
             padding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             axes: tuple[int] | None = None,
            ) -> ndarray:
        """
        Perform a (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The input array.
        `padding` : `FFTPadding` (default: `FFTPadding.NOPADDING`)
            The padding to apply to the array.
        `bc_types` : `tuple[BCType]` or `None` (default: `None`)
            The boundary conditions to apply to each axis.
        `positions` : `tuple[AxisPosition]` or `None` (default: `None`)
            The position of the field.
        `axes` : `tuple[int]` or `None` (default: `None`)
            The axes to transform.
        
        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    @abstractmethod
    def ifft(self, 
             arr: ndarray,
             padding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             axes: tuple[int] | None = None,
             ) -> ndarray:
        """
        Perform an inverse (fast) fourier transform on the input array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The input array.
        `padding` : `FFTPadding` (default: `FFTPadding.NOPADDING`)
            The padding to apply to the array.
        `bc_types` : `tuple[BCType]` or `None` (default: `None`)
            The boundary conditions to apply to each axis.
        `positions` : `tuple[AxisPosition]` or `None` (default: `None`)
            The position of the field.
        `axes` : `tuple[int]` or `None` (default: `None`)
            The axes to transform.
        
        Returns
        -------
        `ndarray`
            The transformed array.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    #  Spectral Analysis Tools
    # ----------------------------------------------------------------

    @abstractmethod
    def omega(self, 
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
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

    @abstractmethod
    def vec_q(self, s: int, use_discrete: bool = True) -> fr.StateBase:
        """
        Computes the eigenvector of the linear operator of the mode `s`.
        
        Parameters
        ----------
        `s` : `int`
            The mode (which eigenvalue / eigenvector to compute).
        `use_discrete` : `bool` (default: True)
            Whether to include space-discretization effects.

        Returns
        -------
        `StateBase`
            The eigenvector of the linear operator.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_p(self, s: int, use_discrete: bool = True) -> fr.StateBase:
        """
        Computes the projection vector of the linear operator of the mode `s`.
        
        Parameters
        ----------
        `s` : `int`
            The mode (which eigenvalue / eigenvector to compute).
        `use_discrete` : `bool` (default: True)
            Whether to include space-discretization effects.

        Returns
        -------
        `StateBase`
            The projection vector of the linear operator.
        """
        raise NotImplementedError

    @property
    def omega_analytical(self) -> ndarray:
        """
        Analytical dispersion relation.
        """
        if self._omega_analytical is None:
            self._omega_analytical = self.omega(self.K, use_discrete=False)
        return self._omega_analytical

    @property
    def omega_space_discrete(self) -> ndarray:
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

    # ----------------------------------------------------------------
    #  Domain Decomposition Methods
    # ----------------------------------------------------------------

    def sync(self, 
             arr: ndarray, 
             flat_axes: list[int] | None = None) -> ndarray:
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
        return self.domain_decomp.sync(arr, flat_axes=flat_axes)

    @fr.utils.jaxjit
    def sync_multi(self, arrs: tuple[ndarray]) -> tuple[ndarray]:
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
        return self.domain_decomp.sync_multiple(arrs)

    @fr.utils.jaxjit
    def unpad(self, arr: ndarray) -> ndarray:
        """
        Remove the halo padding from an array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The padded array.
        
        Returns
        -------
        `ndarray`
            The unpadded array.
        """
        return self.domain_decomp.unpad(arr)

    @fr.utils.jaxjit
    def pad(self, arr: ndarray) -> ndarray:
        """
        Add halo padding to an array.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The unpadded array.
        
        Returns
        -------
        `ndarray`
            The padded array.
        """
        return self.domain_decomp.pad(arr)

    @partial(fr.utils.jaxjit, static_argnames=('pad', 'spectral', 'topo'))
    def create_array(self,
                     pad: bool = True, 
                     spectral: bool = False,
                     topo: tuple[bool] | None = None) -> ndarray:
        """
        Create an array.

        Parameters
        ----------
        `pad` : bool
            Whether to add padding to the array.
        `spectral` : bool
            Whether the array is in spectral space.
        `topo` : tuple[bool] | None
            The topology of the array. Axes with false are flat (only one grid point)
        """
        return self.domain_decomp.create_array(
            pad=pad, spectral=spectral, topo=topo)
    
    def create_random_array(self, 
                            seed: int = 1234,
                            pad: bool = True,
                            spectral: bool = False,
                            topo: tuple[bool] | None = None
                            ) -> ndarray:
        """
        Create a random array.

        Parameters
        ----------
        `seed` : int
            The seed for the random number generator.
        `pad` : bool
            Whether to add padding to the array.
        `spectral` : bool
            Whether the array is in spectral space.
        `topo` : tuple[bool] | None
            The topology of the array. Axes with false are flat (only one grid point)
        """
        return self.domain_decomp.create_random_array(
            seed=seed, pad=pad, spectral=spectral, topo=topo)
    # ----------------------------------------------------------------
    #  Display methods
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

    def __repr__(self) -> str:
        """
        String representation of the grid.
        """
        res = self.name
        for key, value in self.info.items():
            res += "\n  - {}: {}".format(key, value)
        return res

    # ----------------------------------------------------------------
    #  Grid Modules
    # ----------------------------------------------------------------

    @property
    def diff_module(self) -> fr.grid.DiffModule:
        """The differential operator module."""
        return self._diff_module
    
    @diff_module.setter
    def diff_module(self, value: fr.grid.DiffModule) -> None:
        if not isinstance(value, fr.grid.DiffModule):
            raise ValueError("The differential operator module must be a DiffBase object")
        self._diff_module = value
        return
    
    @property
    def interp_module(self) -> fr.grid.InterpolationModule:
        """The interpolation operator module."""
        return self._interp_module
    
    @interp_module.setter
    def interp_module(self, value: fr.grid.InterpolationModule) -> None:
        if not isinstance(value, fr.grid.InterpolationModule):
            raise ValueError("The interpolation operator module must be an InterpolationBase object")
        self._interp_module = value
        return

    @property
    def water_mask(self) -> fr.grid.WaterMask:
        """
        Get the water mask.
        """
        return self._water_mask

    @water_mask.setter
    def water_mask(self, value: fr.grid.WaterMask) -> None:
        self._water_mask = value
        return

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def mset(self) -> fr.ModelSettingsBase | None:
        """The model settings object."""
        return self._mset

    @property
    def domain_decomp(self) -> fr.domain_decomposition.DomainDecomposition:
        """The domain decomposition object."""
        return self._domain_decomp

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
    def cell_center(self) -> fr.grid.Position:
        """The position of the cell centers."""
        return self._cell_center

    @property
    def X(self) -> tuple[ndarray]:
        """The meshgrid of the grid points."""
        return self._X

    @property
    def x_global(self) -> tuple[ndarray]:
        """The x-vector of the global grid points."""
        return self._x_global

    @property
    def K(self) -> ndarray:
        """The wavenumber of the grid."""
        return self._K
    
    @property
    def k_global(self) -> ndarray:
        """The global wavenumber of the grid."""
        return self._k_global
    
    @property
    def dx(self) -> tuple[ndarray]:
        """The grid spacing in each dimension."""
        return self._dx

    @property
    def dV(self) -> ndarray:
        """The volume element of the grid."""
        return self._dV

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
    def spectral_grid(self) -> bool:
        """Indicates whether the grid is a spectral grid."""
        return self._spectral_grid
    
    @spectral_grid.setter
    def spectral_grid(self, value: bool) -> None:
        self._spectral_grid = value
        return
