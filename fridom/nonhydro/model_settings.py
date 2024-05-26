import numpy as np

from fridom.framework.modelsettings_base import ModelSettingsBase

class ModelSettings(ModelSettingsBase):
    """
    Settings container for the model.

    Attributes:
        dtype (np.dtype)       : Data type for real.         
        ctype (np.dtype)       : Data type for complex.       
        gpu (bool)             : Switch for GPU.              
        f0 (dtype)             : Coriolis parameter.         
        beta (dtype)           : Beta term for Coriolis parameter.
        N0 (dtype)             : Brunt-Vaisala frequency.      
        dsqr (dtype)           : Square of aspect ratio.      
        Ro (dtype)             : Rossby number.                
        L (list)               : Domain size in each direction.
        N (list)               : Grid points in each direction.
        dx (dtype)             : Grid spacing in x-direction.
        dy (dtype)             : Grid spacing in y-direction.
        dz (dtype)             : Grid spacing in z-direction.
        periodic_bounds (list) : List of bools for periodic boundaries.
        dt (dtype)             : Time step size.
        eps (dtype)            : 2nd order bashforth correction.
        AB1 (np.ndarray)       : 1st order Adams-Bashforth coefficients.
        AB2 (np.ndarray)       : 2nd order Adams-Bashforth coefficients.
        AB3 (np.ndarray)       : 3rd order Adams-Bashforth coefficients.
        AB4 (np.ndarray)       : 4th order Adams-Bashforth coefficients.

        enable_nonlinear (bool) : Enable nonlinear terms.
        enable_varying_N (bool) : Enable varying stratification.
        enable_varying_f (bool) : Enable varying Coriolis parameter.
        enable_source (bool)    : Enable source terms.
        enable_tqdm (bool)      : Enable progress bar.
        enable_verbose (bool)   : Enable verbose output.

        advection (AdvectionConstructor): Advection scheme.
    """
    def __init__(self, dtype=np.float64, ctype=np.complex128, **kwargs):
        """
        Constructor.

        Args:
            dtype (np.dtype)   : Data type for real.         
            ctype (np.dtype)   : Data type for complex.
        """

        # physical parameters
        self.f0   = dtype(1)
        self.beta = dtype(0)
        self.N0   = dtype(1)
        self.dsqr = dtype(0.2**2)
        self.Ro   = dtype(0.1)

        # ------------------------------------------------------------------
        #   SWITCHES
        
        # Physics
        self.enable_nonlinear  = True    # Enable nonlinear terms
        self.enable_varying_N  = False   # Enable varying stratification
        self.enable_varying_f  = False   # Enable varying Coriolis parameter
        self.enable_source     = False   # Enable source terms

        # Advection
        from fridom.nonhydro.modules.advection.second_order_advection import SecondOrderAdvectionConstructor
        self.advection = SecondOrderAdvectionConstructor()

        # init function must be called after all new variables are set
        super().__init__(n_dims=3, dtype=dtype, ctype=ctype, **kwargs)

        # Some parameters would be overwritten by the init function
        # so we set them again here
        self.model_name = "NonHydrostatic"
        return


    def __str__(self) -> str:
        """
        String representation of the model settings.

        Returns:
            res (str): String representation of the model settings.
        """
        res = super().__str__()
        res += "  Physical parameters:\n"
        res += "    f0   = {:.3f}\n".format(self.f0)
        res += "    beta = {:.3f}\n".format(self.beta)
        res += "    N0   = {:.3f}\n".format(self.N0)
        res += "    dsqr = {:.3f}\n".format(self.dsqr)
        res += "    Ro   = {:.3f}\n".format(self.Ro)
        res += "{}".format(self.advection)
        res += "  Switches:\n"
        res += "    enable_nonlinear  = {}\n".format(self.enable_nonlinear)
        res += "    enable_varying_N  = {}\n".format(self.enable_varying_N)
        res += "    enable_varying_f  = {}\n".format(self.enable_varying_f)
        res += "    enable_source     = {}\n".format(self.enable_source)
        res += "================================================\n"
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
        self.max_cg_iter = max(self._N)
    
    @property
    def dx(self) -> float:
        """Grid spacing in x-direction."""
        return self.dg[0]
    
    @property
    def dy(self) -> float:
        """Grid spacing in y-direction."""
        return self.dg[1]
    
    @property
    def dz(self) -> float:
        """Grid spacing in z-direction."""
        return self.dg[2]

# remove symbols from namespace
del np, ModelSettingsBase