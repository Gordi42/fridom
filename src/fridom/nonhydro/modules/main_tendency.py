import fridom.framework as fr
import fridom.nonhydro as nh


class MainTendency(fr.modules.ModuleContainer):
    def __init__(self,
                 name="All Tendency Modules",
                 linear_tendency=None,
                 advection=None,
                 tendency_divergence=None,
                 pressure_solver=None,
                 pressure_gradient_tendency=None):
        mods = nh.modules
        if linear_tendency is None:
            linear_tendency = mods.LinearTendency()
        if advection is None:
            advection = mods.advection.CenteredAdvection()
        if tendency_divergence is None:
            tendency_divergence = mods.TendencyDivergence()
        if pressure_solver is None:
            pressure_solver = mods.pressure_solvers.SpectralPressureSolver()
        if pressure_gradient_tendency is None:
            pressure_gradient_tendency = mods.PressureGradientTendency()

        module_list = [
            linear_tendency,             # Always on element 0
            advection,                   # Always on element 1
            tendency_divergence,         # Always on element -3
            pressure_solver,             # Always on element -2
            pressure_gradient_tendency,  # Always on element -1
        ]
        super().__init__(name=name, module_list=module_list)
        self.additional_modules = []
        return

    def add_module(self, module):
        # add the module to the additional_modules list
        self.additional_modules.append(module)
        # update the module list
        # we need to make sure that the pressure solver and the pressure 
        # gradient tendency are always in the last two positions
        module_list = []
        module_list.append(self.linear_tendency)
        module_list.append(self.advection)
        module_list += self.additional_modules
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
        return self.module_list[0]
    
    @linear_tendency.setter
    def linear_tendency(self, value):
        self.module_list[0] = value
        return
    
    @property
    def advection(self):
        return self.module_list[1]
    
    @advection.setter
    def advection(self, value):
        self.module_list[1] = value
        return

    @property
    def tendency_divergence(self):
        return self.module_list[-3]
    
    @tendency_divergence.setter
    def tendency_divergence(self, value):
        self.module_list[-3] = value
        return

    @property
    def pressure_solver(self):
        return self.module_list[-2]
    
    @pressure_solver.setter
    def pressure_solver(self, value):
        self.module_list[-2] = value
        return
    
    @property
    def pressure_gradient_tendency(self):
        return self.module_list[-1]
    
    @pressure_gradient_tendency.setter
    def pressure_gradient_tendency(self, value):
        self.module_list[-1] = value
        return