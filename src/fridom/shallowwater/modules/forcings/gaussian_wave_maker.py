from fridom.shallowwater.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class GaussianWaveMaker(Module):
    """
    A Gaussian wave maker that forces the u-component of the velocity field.
    """

    def __init__(self, 
                 position: tuple, 
                 width: tuple, 
                 frequency: float,
                 amplitude: float):
        """
        Constructor of the wave maker source term.
        Adds an unpolarized gaussian signal to the h-component.

        ## Arguments:
            position (tuple):                   position of the source term
            width (tuple):                      width of the source term
            frequency (float):                  frequency of the source term
            amplitude (float):                  amplitude of the source term
        """
        super().__init__(name="Gaussian Wave Maker",
                         position=position,
                         width=width,
                         frequency=frequency,
                         amplitude=amplitude)

    @start_module
    def start(self):
        # shorthand
        cp = self.grid.cp
        X, Y = tuple(self.grid.X)

        # Create gaussian mask
        self.mask = cp.exp(-((X - self.position[0])**2 / self.width[0]**2 +
                             (Y - self.position[1])**2 / self.width[1]**2 ) )
        self.mask *= self.amplitude
        return

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Update the state with the source term.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        cp = self.grid.cp
        dz.h += self.mask * cp.sin(2 * cp.pi * self.frequency * mz.clock.time)
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    position: {self.position}\n"
        res += f"    width: {self.width}\n"
        res += f"    frequency: {self.frequency}\n"
        res += f"    amplitude: {self.amplitude}\n"
        return res


# remove symbols from namespace
del State, ModelState, Module, update_module, start_module