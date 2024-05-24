from abc import abstractmethod

from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State

class Source(State):
    """
    Parent class for source terms in the model.

    ## Methods:
        update:     Function that is called every time step
                    to update the source term.
    """
    def __init__(self, grid: Grid):
        super().__init__(grid)
        return

    @abstractmethod
    def update(self, time: float):
        """
        Dummy function to be overridden by child classes.
        Function is called every time step to update the source term.

        ## Arguments:
            time (float):   current time in simulation
        """
        return


class WaveMaker(Source):
    """
    Class for the wavemaker source term.
    """
    def __init__(self, grid: Grid,
                 position: tuple, width: tuple, frequency: float,
                 amplitude: float):
        """
        Constructor of the wave maker source term.
        Adds an unpolarized gaussian signal to the model.

        ## Arguments:
            grid (Grid):                        object containing grid information
            position (tuple):                   position of the source term
            width (tuple):                      width of the source term
            frequency (float):                  frequency of the source term
            amplitude (float):                  amplitude of the source term
        """
        super().__init__(grid)

        # shorthand
        cp = self.cp
        # Create gaussian mask
        mask = cp.ones_like(grid.X[0])
        for x, pos, width in zip(grid.X, position, width):
            mask *= cp.exp(-(x - pos)**2 / width**2)

        # Store parameters
        self.position  = position
        self.width     = width
        self.frequency = frequency
        self.amplitude = amplitude
        self.mask      = mask
        return

    def update(self, time: float):
        """
        Function that is called every time step
        to update the source term.

        ## Arguments:
            time (float):   current time in simulation
        """
        cp   = self.cp
        amp  = self.amplitude
        freq = self.frequency
        mask = self.mask
        self.u[:] = amp * mask * cp.sin(2 * cp.pi * freq * time)
        return


class PolarizedWaveMaker(Source):
    """
    Class for the polarized wavemaker source term.
    """
    def __init__(self, grid: Grid,
                 kx=6, ky=0, s=1, amplitude=1,
                 mask_pos=(0.5, 0.5), mask_width=(0.2,0.2)) -> None:
        """
        Constructor of the SingleWave initial condition.

        Arguments:
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            kz (float)            : The wavenumber in the z-direction.
            s (int)               : The mode (1, -1)
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            mask_pos (tuple)      : The position of the mask.
            mask_width (tuple)    : The width of the mask.
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp


        # Construct the polarized wave
        from fridom.shallowwater.initial_conditions import SingleWave
        z = SingleWave(grid, kx, ky, s)
        self.omega = z.omega.real

        # set mask
        mask = cp.ones_like(grid.X[0])
        for x, pos, width in zip(grid.X, mask_pos, mask_width):
            if pos is not None and width is not None:
                mask *= cp.exp(-(x - pos)**2 / width**2)

        mask *= amplitude

        z.u *= mask
        z.v *= mask
        z.h *= mask

        self.z_mask = z.copy()

        # project again on wave mode
        from fridom.shallowwater.eigenvectors import VecQ, VecP
        q = VecQ(s, grid)
        p = VecP(s, grid)
        z = z.fft()
        proj = z.dot(p)
        z_real = (q*proj).fft()
        z_imag = (q*(proj*(-1j))).fft()

        # save the state
        self.z_real = z_real
        self.z_imag = z_imag
        
        return
    
    def update(self, time: float):
        """
        Function that is called every time step
        to update the source term.

        ## Arguments:
            time (float):   current time in simulation
        """
        cp = self.cp
        phase = time * self.omega
        z = self.z_real * cp.cos(phase) + self.z_imag * cp.sin(phase)

        self.u[:] = z.u
        self.v[:] = z.v
        self.h[:] = z.h
        return


# remove symbols from namespace
del abstractmethod, Grid, State