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

        super().__init__(grid, State, ModelState)
        self.mset = grid.mset
        return


# remove symbols from namespace
del Grid, ModelBase