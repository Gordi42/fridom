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
            LinearTendency, PressureGradientTendency, TendencyDivergence
        from fridom.nonhydro.modules.pressure_solvers import SpectralPressureSolver
        from fridom.nonhydro.modules.advection import SecondOrderAdvection

        mset = grid.mset
        super().__init__(grid, State, ModelState)
        self.mset = mset
        
        # Modules
        self.linear_tendency     = LinearTendency()
        self.advection           = SecondOrderAdvection()
        self.tendency_divergence = TendencyDivergence()
        self.pressure_gradient   = PressureGradientTendency()
        self.pressure_solver     = SpectralPressureSolver()

        self.linear_tendency.start(grid=grid, timer=self.timer)
        self.advection.start(grid=grid, timer=self.timer)
        self.tendency_divergence.start(grid=grid, timer=self.timer)
        self.pressure_gradient.start(grid=grid, timer=self.timer)
        self.pressure_solver.start(grid=grid, timer=self.timer)

        return

    # ============================================================
    #   TOTAL TENDENCY
    # ============================================================

    def total_tendency(self):
        
        # calculate linear tendency
        self.linear_tendency.update(self.model_state, self.dz)

        # calculate nonlinear tendency
        if self.mset.enable_nonlinear:
            self.advection.update(self.model_state, self.dz)

        # solve for pressure
        self.tendency_divergence.update(self.model_state, self.dz)
        self.pressure_solver.update(self.model_state, self.dz)

        # calculate pressure gradient tendency
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

# remove symbols from namespace
del Grid, ModelBase