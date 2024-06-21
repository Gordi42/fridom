import numpy as np


class ModelSettingsBase:
    """
    Base class for model settings container.
    
    Description
    -----------
    This class should be used as a base class for all model settings containers.
    It provides a set of attributes and methods that are common to all models.
    Child classes should override the following attributes: 
    - n_dims 
    - model_name
    - tendencies
    - diagnostics
    - state_constructor
    - diagnostic_state_constructor

    
    Attributes
    ----------
    `dtype` : `np.dtype` (default: `np.float64`)
        The data type for real numbers.
    `ctype` : `np.dtype` (default: `np.complex128`)
        The data type for complex numbers.
    `model_name` : `str` (default: `"Unnamed model"`)
        The name of the model.
    `gpu` : `bool` (default: `True` if `cupy` is installed)
        Switch for GPU acceleration.
    `n_dims` : `int`
        The number of spatial dimensions.
    `L` : `list[float]`
        The domain size in each direction.
    `N` : `list[int]`
        The number of grid points in each direction.
    `dg` : `list[float]`
        The grid spacing in each direction. (read-only)
    `total_grid_points` : `int` (read-only)
        The total number of grid points.
    `periodic_bounds` : `list[bool]`
        A list of booleans indicating whether the boundaries are periodic.
    `time_stepper` : `TimeStepperBase` (default: `AdamBashforth()`
        The time stepper object.
    `tendencies` : `ModuleContainer`
        A container for all modules that calculate tendencies.
    `diagnostics` : `ModuleContainer`
        A container for all modules that calculate diagnostics.
    `state_constructor` : `callable`
        A function that constructs a state from the grid.
    `diagnostic_state_constructor` : `callable`
        A function that constructs a diagnostic state from the grid.
    `enable_tqdm` : `bool` (default: `True`)
        Enable progress bar.
    `enable_verbose` : `bool` (default: `False`)
        Enable verbose output.
    
    
    Methods
    -------
    `copy()`
        Returns a deep copy of the model settings.
    `print_verbose(*args, **kwargs)`
        Print function for verbose output.
    `set_attributes(**kwargs)`
        Set attributes from keyword arguments.
    
    Examples
    --------
    Create a new model settings class by inheriting from `ModelSettingsBase`:

    >>> from fridom.framework import ModelSettingsBase
    >>> class ModelSettings(ModelSettingsBase):
    ...     def __init__(self, dtype=np.float64, ctype=np.complex128, **kwargs):
    ...         super().__init__(n_dims=2, dtype=dtype, ctype=ctype)
    ...         self.model_name = "MyModel"
    ...         # make default domain size 2*pi x 2*pi with 63 x 63 grid points
    ...         self.L = [2*np.pi, 2*np.pi]
    ...         self.N = [63, 63]
    ...         # set other parameters
    ...         self.my_parameter = 1.0
    ...         # set up modules and state constructors here. Eee for example 
    ...         # the `ModelSettings` class in `shallowwater/model_settings.py`
    ...         # Finally, set attributes from keyword arguments
    ...         self.set_attributes(**kwargs)
    ...     # maybe update the __str__ method to include the new parameter
    ...     def __str__(self) -> str:
    ...         res = super().__str__()
    ...         res += "  My parameter: {}\\n".format(self.my_parameter)
    >>> settings = ModelSettings(L=[4*np.pi, 4*np.pi], my_parameter=2.0)
    >>> print(settings)
    """
    def __init__(self, n_dims:int, dtype=np.float64, ctype=np.complex128,
                 **kwargs) -> None:
        """
        Constructor of the model settings base class. Should be called by the
        constructor of the child class.
        
        Parameters
        ----------
        `n_dims` : `int`
            The number of spatial dimensions.
        `dtype` : `np.dtype` (default: `np.float64`)
            The data type for real numbers.
        `ctype` : `np.dtype` (default: `np.complex128`)
            The data type for complex numbers.
        
        Examples
        --------
        See the examples in the class docstring
        """

        # variable types
        self.dtype = dtype
        self.ctype = ctype

        # model name
        self.model_name = "Unnamed model"

        # GPU acceleration
        self.__gpu = True
        try:
            import cupy
        except ImportError: self.__gpu = False

        # spatial parameters
        self.n_dims = n_dims
        self._L  = [dtype(1)] * n_dims
        self._N  = [64] * n_dims
        self._total_grid_points = 64**n_dims
        self._dg = [L / N for L, N in zip(self._L, self._N)]

        # Boundary conditions
        self.periodic_bounds = [True] * n_dims

        # time parameters
        from fridom.framework.time_steppers.adam_bashforth import AdamBashforth
        self.time_stepper = AdamBashforth()

        # modules
        from fridom.framework.modules.module_container import ModuleContainer
        # List of modules that calculate tendencies
        self.tendencies = ModuleContainer(name="All Tendency Modules")
        # List of modules that do diagnostics
        self.diagnostics = ModuleContainer(name="All Diagnostic Modules")

        # ------------------------------------------------------------------
        #   SWITCHES
        # ------------------------------------------------------------------

        # Output
        self.enable_tqdm       = True    # Enable progress bar
        self.enable_verbose    = False   # Enable verbose output

        # State Constructors
        from fridom.framework.state_base import StateBase
        self.state_constructor = lambda grid: StateBase(grid, [])
        self.diagnostic_state_constructor = lambda grid: StateBase(grid, [])

        # Set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def set_attributes(self, **kwargs):
        """
        Set model settings attributes from keyword arguments. If an attribute
        does not exist, an AttributeError is raised.
        
        Parameters
        ----------
        `**kwargs` : `dict`
            Keyword arguments to set the attributes of the model settings.
        
        Raises
        ------
        `AttributeError`
            The attribute does not exist in the model settings.
        """
        # Set attributes from keyword arguments
        for key, value in kwargs.items():
            # Check if attribute exists
            if not hasattr(self, key):
                raise AttributeError(
                    "ModelSettings has no attribute '{}'".format(key)
                    )
            setattr(self, key, value)
        return

    def copy(self):
        """
        Return a deepcopy of the model settings.
        """
        import copy
        return copy.deepcopy(self)

    def print_verbose(self, *args, **kwargs):
        """
        Print only if verbose output is enabled.
        """
        if self.enable_verbose:
            print(*args, **kwargs)
        return

    def __str__(self) -> str:
        """
        String representation of the model settings.
        """
        res = "================================================\n"
        res += "  Model Settings:\n"
        res += "================================================\n"
        res += "  Model name: {}\n".format(self.model_name)
        res += "  Spatial parameters:\n"
        res += "    n_dims = {}\n".format(self.n_dims)
        res += "    L = [{:.3f}".format(self.L[0])
        for i in range(1, self.n_dims):
            res += ", {:.3f}".format(self.L[i])
        res += "]\n"
        res += "    N = {}\n".format(self.N)
        res += "    dg = [{:.3f}".format(self.dg[0])
        for i in range(1, self.n_dims):
            res += ", {:.3f}".format(self.dg[i])
        res += "]\n"
        res += "  Boundary conditions:\n"
        res += "    Periodic : {}\n".format(self.periodic_bounds)
        res += f"{self.time_stepper}"
        res += f"{self.tendencies}"
        res += f"{self.diagnostics}"
        res += "------------------------------------------------\n"
        return res

    def __repr__(self) -> str:
        """
        String representation of the model settings (for IPython).
        """
        return self.__str__()


    # ==================================================================
    #  GETTER AND SETTER FOR PRIVATE VARIABLES
    # ==================================================================
    @property
    def L(self) -> list:
        """Domain size in each direction."""
        return self._L

    @L.setter
    def L(self, value: list):
        self._L = [self.dtype(val) for val in value]
        self._dg = [L / N for L, N in zip(self._L, self._N)]    

    @property
    def N(self) -> list:
        """Grid points in each direction."""
        return self._N
    
    @N.setter
    def N(self, value: list):
        self._N = [int(val) for val in value]
        self._dg = [L / N for L, N in zip(self._L, self._N)]
        self._total_grid_points = 1
        for n in self._N:
            self._total_grid_points *= n

    @property
    def dg(self) -> list:
        """Grid spacing in each direction."""
        return self._dg

    @property
    def total_grid_points(self) -> list:
        """Total number of grid points."""
        return self._total_grid_points

    @property
    def gpu(self) -> bool:
        """Switch for GPU acceleration."""
        return self.__gpu
    
    @gpu.setter
    def gpu(self, value: bool):
        if value:
            try: 
                import cupy
            except ImportError: 
                print("WARNING: Cupy is not installed. GPU acceleration is not available.")
                self.__gpu = False
        else:
            self.__gpu = value
        return
