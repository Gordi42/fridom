import fridom.framework as fr
import fridom.nonhydro as nh


class MainTendency(fr.modules.ModuleContainer):
    name = "Main Tendencies: Nonhydrostatic Model"
    def __init__(self):
        mods = nh.modules
        self._reset_tendency = mods.ResetTendency()
        self._linear_tendency = mods.LinearTendency()
        self._advection = mods.advection.CenteredAdvection()
        self._tendency_divergence = mods.TendencyDivergence()
        self._pressure_solver = mods.pressure_solvers.SpectralPressureSolver()
        self._pressure_gradient_tendency = mods.PressureGradientTendency()
        self._additional_modules = []
        self._set_module_list()

        super().__init__(module_list=self.module_list)
        return

    def add_module(self, module):
        self._additional_modules.append(module)
        self._set_module_list()
        return

    def _set_module_list(self):
        """
        Set the module list.
        
        Description
        -----------
        This function make sure that the pressure solver and the pressure
        gradient tendency are always in the last two positions.
        """
        module_list = []
        module_list.append(self._reset_tendency)
        module_list.append(self.linear_tendency)
        module_list.append(self.advection)
        module_list += self._additional_modules
        module_list.append(self.tendency_divergence)
        module_list.append(self.pressure_solver)
        module_list.append(self.pressure_gradient_tendency)
        self.module_list = module_list
        return

    # ============================================================
    #   PROPERTIES
    # ============================================================

    @property
    def linear_tendency(self):
        """The core linear momentum tendency module."""
        return self._linear_tendency
    
    @linear_tendency.setter
    def linear_tendency(self, value):
        self._linear_tendency = value
        self._set_module_list()
        return
    
    @property
    def advection(self):
        """The advection module (nonlinear + linear by backgound state)."""
        return self._advection
    
    @advection.setter
    def advection(self, value):
        self._advection = value
        self._set_module_list()
        return

    @property
    def tendency_divergence(self):
        """The divergence of the momentum tendency, for the pressure solver"""
        return self._tendency_divergence
    
    @tendency_divergence.setter
    def tendency_divergence(self, value):
        self._tendency_divergence = value
        self._set_module_list()
        return

    @property
    def pressure_solver(self):
        """The pressure solver module."""
        return self._pressure_solver
    
    @pressure_solver.setter
    def pressure_solver(self, value):
        self._pressure_solver = value
        self._set_module_list()
        return
    
    @property
    def pressure_gradient_tendency(self):
        """The pressure gradient tendency module."""
        return self._pressure_gradient_tendency
    
    @pressure_gradient_tendency.setter
    def pressure_gradient_tendency(self, value):
        self._pressure_gradient_tendency = value
        self._set_module_list()
        return