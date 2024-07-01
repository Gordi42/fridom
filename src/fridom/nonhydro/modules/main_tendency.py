# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules.module_container import ModuleContainer
from fridom.framework.modules.module import Module
from fridom.nonhydro.modules.linear_tendency import LinearTendency
from fridom.nonhydro.modules.advection \
    .second_order_advection import SecondOrderAdvection
from fridom.nonhydro.modules.tendency_divergence import TendencyDivergence
from fridom.nonhydro.modules \
    .pressure_gradient_tendency import PressureGradientTendency
from fridom.nonhydro.modules.pressure_solvers \
    .spectral_pressure_solver import SpectralPressureSolver
# Import type information
if TYPE_CHECKING:
    pass



class MainTendency(ModuleContainer):
    def __init__(self,
                 name="All Tendency Modules",
                 linear_tendency=LinearTendency(),
                 advection=Module(name="None"),
                 tendency_divergence=TendencyDivergence(),
                 pressure_solver=SpectralPressureSolver(),
                 pressure_gradient_tendency=PressureGradientTendency()):
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