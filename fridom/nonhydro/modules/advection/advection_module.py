from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State
from fridom.framework.timing_module import TimingModule

class AdvectionModule:
    """
    Base class for advection modules. This class defines the interface for the 
    advection term of the state vector z.
    """
    def __init__(self, 
                 grid: Grid,
                 timer: TimingModule):
        self.grid = grid
        self.mset = grid.mset
        self.timer = timer

    def __call__(self, 
                 z: State, 
                 dz: State,
                 z_background: State = None) -> None:
        """
        Compute the advection term of the state vector z.

        Args:
            z (State)  : State object.
            dz (State) : Advection term of the state.
            z_background (State) : Background state object (optional).
        """
        # start the timer
        self.timer.get("Advection").start()

        # compute the advection term
        self.compute_advection(z, dz, z_background)

        # stop the timer
        self.timer.get("Advection").stop()
        return

    def compute_advection(self,
                          z: State,
                          dz: State,
                          z_background: State = None) -> None:
            """
            Compute the advection term of the state vector z.
            This function has to be implemented by the derived class.
    
            Args:
                z (State)  : State object.
                dz (State) : Advection term of the state.
                z_background (State) : Background state object (optional).
            """
            raise NotImplementedError


class AdvectionConstructor:
    """
    This class provides a constructor for advection modules.
    All advection modules should have a __call__ method that takes the
    ModelSettings, Grid, and TimingModule as parameters and returns the advection module.
    """

    def __call__(self, 
                 grid: Grid,
                 timer: TimingModule) -> AdvectionModule:
        """
        This function returns the advection module.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "  Advection Scheme: \n    undefined\n"


# remove symbols from the namespace
del Grid, State, TimingModule