import numpy as np


class ModelSettingsBase:
    """
    Base class for model settings container. 

    Attributes:
        dtype (np.dtype)       : Data type for real.         
        ctype (np.dtype)       : Data type for complex.       
        model_name (str)       : Name of the model.
        gpu (bool)             : Switch for GPU.              
        n_dims (int)           : Number of spatial dimensions.
        L (list)               : Domain size in each direction.
        N (list)               : Grid points in each direction.
        dg (list)              : Grid spacing in each direction.
        periodic_bounds (list) : List of bools for periodic boundaries.

        enable_tqdm (bool)      : Enable progress bar.
        enable_verbose (bool)   : Enable verbose output.

    Methods:
        copy           : Return a copy of the model settings.
        print_verbose  : Print function for verbose output.
        __str__        : String representation of the model settings.
        __repr__       : String representation of the model settings (for 
                         IPython).
    """

    def __init__(self, n_dims:int, dtype=np.float64, ctype=np.complex128,
                 **kwargs):
        """
        Constructor.

        Args:
            n_dims (int)       : Number of spatial dimensions.
            dtype (np.dtype)   : Data type for real.         
            ctype (np.dtype)   : Data type for complex.
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
        self.state_constructor = lambda grid: StateBase(grid)
        from fridom.framework.model_state import ModelStateBase
        self.model_state_constructor = lambda grid: ModelStateBase(grid)


        # Set attributes from keyword arguments
        for key, value in kwargs.items():
            # Check if attribute exists
            if not hasattr(self, key):
                raise AttributeError(
                    "ModelSettings has no attribute '{}'".format(key)
                    )
            setattr(self, key, value)

    def copy(self):
        """
        Return a copy of the model settings.
        """
        import copy
        return copy.deepcopy(self)

    def print_verbose(self, *args, **kwargs):
        """
        Print function for verbose output.
        """
        if self.enable_verbose:
            print(*args, **kwargs)
        return

    def __str__(self) -> str:
        """
        String representation of the model settings.

        Returns:
            res (str): String representation of the model settings.
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

        Returns:
            res (str): String representation of the model settings.
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
