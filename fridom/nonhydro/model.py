from fridom.nonhydro.grid import Grid
from fridom.framework.model_base import ModelBase
from fridom.nonhydro.model_state import ModelState


class Model(ModelBase):
    """
    A 3D non-hydrostatic Boussinesq model. Based on the ps3D model by
    Prof. Carsten Eden ( https://github.com/ceden/ps3D ).

    Attributes:
        z (State)               : State variables (u,v,w,b).
        p (FieldVariable)       : Pressure (p).
        it (int)                : Iteration counter.
        timer (TimingModule)    : Timer.
        Modules:                : Modules of the model.

    Methods:
        step()                  : Perform one time step.
        run()                   : Run the model for a given number of steps.
        reset()                 : Reset the model (pointers, tendencies)
        update_pointer()        : Update pointer for Adam-Bashforth time stepping.
        update_coeff_AB()       : Update coeffs for Adam-Bashforth time stepping.
        adam_bashforth()        : Perform Adam-Bashforth time stepping.
    """

    def __init__(self, grid:Grid) -> None:
        """
        Constructor.

        Args:
            grid (Grid)             : Grid.
        """
        # import Modules
        from fridom.nonhydro.state import State
        from fridom.nonhydro.modules import \
            LinearTendency, PressureGradientTendency, PressureSolve, \
            SourceTendency

        mset = grid.mset
        super().__init__(grid, State, ModelState)
        self.mset = mset
        
        # Modules
        self.linear_tendency     = LinearTendency()
        self.linear_tendency.start(grid=grid, timer=self.timer)
        self.advection           = mset.advection(grid, self.timer)
        self.pressure_gradient   = PressureGradientTendency()
        self.pressure_gradient.start(grid=grid, timer=self.timer)
        self.pressure_solver     = PressureSolve(grid, self.timer)
        self.source_tendency     = SourceTendency(grid, self.timer)


        return

    # ============================================================
    #   TOTAL TENDENCY
    # ============================================================

    def total_tendency(self):
        
        # calculate linear tendency
        self.linear_tendency.update(self.model_state, self.dz)

        # calculate nonlinear tendency
        if self.mset.enable_nonlinear:
            self.advection(self.z, self.dz)

        if self.mset.enable_source:
            self.source_tendency(self.dz, self.time)

        # solve for pressure
        self.pressure_solver(self.dz, self.p)
        self.pressure_gradient.update(self.model_state, self.dz)

        return



    # ============================================================
    #   OTHER METHODS
    # ============================================================

    def reset(self) -> None:
        """
        Reset the model (pointers, tendencies).
        """
        super().reset()
        self.p = self.p*0
        return

    # ============================================================
    #  to be deleted once the modules are implemented
    # ============================================================
    @property
    def p(self):
        return self.model_state.p
    
    @p.setter
    def p(self, p):
        self.model_state.p = p
        return
    
    @property
    def div(self):
        return self.model_state.div
    
    @div.setter
    def div(self, div):
        self.model_state.div = div
        return

# remove symbols from namespace
del Grid, ModelBase