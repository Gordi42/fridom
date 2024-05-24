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
        ahbi (dtype)           : Hor. biharmonic friction coeff.
        avbi (dtype)           : Ver. biharmonic friction coeff.
        khbi (dtype)           : Hor. biharmonic mixing coeff.
        kvbi (dtype)           : Ver. biharmonic mixing coeff.
        ah (dtype)             : Hor. harmonic friction coeff.
        av (dtype)             : Ver. harmonic friction coeff.
        kh (dtype)             : Hor. harmonic mixing coeff.
        kv (dtype)             : Ver. harmonic mixing coeff.
        dt (dtype)             : Time step size.
        eps (dtype)            : 2nd order bashforth correction.
        AB1 (np.ndarray)       : 1st order Adams-Bashforth coefficients.
        AB2 (np.ndarray)       : 2nd order Adams-Bashforth coefficients.
        AB3 (np.ndarray)       : 3rd order Adams-Bashforth coefficients.
        AB4 (np.ndarray)       : 4th order Adams-Bashforth coefficients.
        snap_interval (int)    : Snapshot interval.
        diag_interval (int)    : Diagnostic interval.
        snap_filename (str)    : Snapshot filename.

        enable_nonlinear (bool) : Enable nonlinear terms.
        enable_varying_N (bool) : Enable varying stratification.
        enable_varying_f (bool) : Enable varying Coriolis parameter.
        enable_source (bool)    : Enable source terms.
        enable_biharmonic (bool): Enable biharmonic friction and mixing.
        enable_harmonic (bool)  : Enable harmonic friction and mixing.
        enable_tqdm (bool)      : Enable progress bar.
        enable_snap (bool)      : Enable writing snapshots.
        enable_diag (bool)      : Enable diagnostic output.
        enable_verbose (bool)   : Enable verbose output.

        enable_live_anim  (bool): Enable live animation.
        live_plot_interval(int) : Live plot interval.
        live_plotter      (cls) : Live plotter class.
        enable_vid_anim   (bool): Enable mp4 animation.
        vid_anim_interval (int) : Video animation interval.
        vid_anim_filename (str) : Video animation filename.
        vid_plotter       (cls) : Video plotter class.
        vid_fps           (int) : Video frames per second.

        pressure_solver (str)   : Choose from "Spectral" or "CG".
        max_cg_iter (int)       : Maximum number of CG iterations.
        cg_tol (float)          : CG tolerance.
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

        # friction and mixing parameters
        self.ahbi = dtype(0)
        self.avbi = dtype(0)
        self.khbi = dtype(0)
        self.kvbi = dtype(0)
        self.ah   = dtype(0)
        self.av   = dtype(0)
        self.kh   = dtype(0)
        self.kv   = dtype(0)

        # ------------------------------------------------------------------
        #   SWITCHES
        
        # Physics
        self.enable_nonlinear  = True    # Enable nonlinear terms
        self.enable_varying_N  = False   # Enable varying stratification
        self.enable_varying_f  = False   # Enable varying Coriolis parameter
        self.enable_source     = False   # Enable source terms
        self.enable_biharmonic = False   # Enable biharmonic friction and mixing
        self.enable_harmonic   = False   # Enable harmonic friction and mixing

        # Pressure solver
        self.pressure_solver   = "Spectral" # Choose from "Spectral" or "CG"
        self.max_cg_iter       = None
        self.cg_tol            = 1e-10      # Conjugate gradient tolerance

        # Advection
        from fridom.nonhydro.modules.advection.second_order_advection import SecondOrderAdvectionConstructor
        self.advection = SecondOrderAdvectionConstructor()

        # init function must be called after all new variables are set
        super().__init__(n_dims=3, dtype=dtype, ctype=ctype, **kwargs)

        # Some parameters would be overwritten by the init function
        # so we set them again here
        self.model_name = "NonHydrostatic"
        if "max_cg_iter" in kwargs:
            self.max_cg_iter   = kwargs["max_cg_iter"]
        else:
            self.max_cg_iter       = max(self.N)
        return


    def scale_biharmonic(self):
        """
        Scale biharmonic friction and mixing coefficients with grid spacing.
        """
        self.ahbi = self.dtype(self.Ro/2 * self.dx**4)
        self.avbi = self.dtype(self.Ro/2 * self.dz**4)
        self.khbi = self.dtype(self.Ro/2 * self.dx**4)
        self.kvbi = self.dtype(self.Ro/2 * self.dz**4)
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
        res += "  Friction and mixing parameters:\n"
        res += "    ahbi = {:.2e}".format(self.ahbi)
        res += "    avbi = {:.2e}\n".format(self.avbi)
        res += "    khbi = {:.2e}".format(self.khbi)
        res += "    kvbi = {:.2e}\n".format(self.kvbi)
        res += "    ah   = {:.2e}".format(self.ah)
        res += "    av   = {:.2e}\n".format(self.av)
        res += "    kh   = {:.2e}".format(self.kh)
        res += "    kv   = {:.2e}\n".format(self.kv)
        res += "  Pressure solver:\n"
        res += "    pressure_solver = {}\n".format(self.pressure_solver)
        res += "    max_cg_iter     = {}\n".format(self.max_cg_iter)
        res += "    cg_tol          = {}\n".format(self.cg_tol)
        res += "{}".format(self.advection)
        res += "  Switches:\n"
        res += "    enable_nonlinear  = {}\n".format(self.enable_nonlinear)
        res += "    enable_varying_N  = {}\n".format(self.enable_varying_N)
        res += "    enable_varying_f  = {}\n".format(self.enable_varying_f)
        res += "    enable_source     = {}\n".format(self.enable_source)
        res += "    enable_biharmonic = {}\n".format(self.enable_biharmonic)
        res += "    enable_harmonic   = {}\n".format(self.enable_harmonic)
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