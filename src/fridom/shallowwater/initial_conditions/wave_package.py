from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class WavePackage(State):
    """
    A single wave package.
    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, grid:Grid, 
                 kx=6, ky=0, s=1, phase=0, 
                 mask_pos=(0.5, 0.5), mask_width=(0.2, 0.2)) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (real)          : The phase of the wave. (Default: 0)
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp

        # Construct single wave
        from fridom.shallowwater.initial_conditions \
            .single_wave import SingleWave
        z = SingleWave(grid, kx, ky, s, phase)

        if not s == 0:
            self.omega = z.omega
            self.period = z.period

        # Construct mask
        mask = cp.ones_like(grid.X[0])
        for x, pos, width in zip(grid.X, mask_pos, mask_width):
            if pos is not None and width is not None:
                mask *= cp.exp(-(x - pos)**2 / width**2)

        z.u *= mask
        z.v *= mask
        z.h *= mask

        # Project onto the mode again
        from fridom.shallowwater.eigenvectors import VecQ, VecP
        q = VecQ(s, grid)
        p = VecP(s, grid)

        z = z.project(p, q)

        if not s == 0:
            z *= 2

        # save the state
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h

        return


# remove symbols from namespace
del Grid, State
