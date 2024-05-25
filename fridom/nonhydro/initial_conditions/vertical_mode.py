from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State


class VerticalMode(State):
    """
    An initial condition that consist of a single verticle mode with a 
    given wavenumber and a given mode.

    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        kz (float)     : The wavenumber in the z-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, grid:Grid, 
                 kx=6, ky=0, kz=4, s=1, phase=0, use_discrete=True) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            kz (float)            : The wavenumber in the z-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (real)          : The phase of the wave. (Default: 0)
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp
        mset = grid.mset

        # Note that the fourier transformation assumes that the first value
        # is at z=0 and the last value is at z=(H-\Delta z). However, due to
        # the grid structure, the first value is at z=\Delta z and the last
        # value is at z=H. To satisfy the boundary condition w(z=H)=0, we
        # need to shift the values by one grid point. This can be done by 
        # adding a phase shift corresponding to a shift of one grid point.

        # Super position of two waves with opposite phase and opposite kz
        from fridom.nonhydro.initial_conditions import SingleWave
        z1 = SingleWave(grid, kx, ky, kz, s, phase)
        
        # calculate the phase shift of the first wave that corresponds to one
        # grid point.
        delta_phase_z1 = mset.dz * z1.kz
        # the phase shift of the second wave is the same but with opposite sign.
        # instead of shifting both waves by the same amount, we shift the second
        # wave by the difference of the two phase shifts.
        phase_shift_z2 = cp.pi - 2 * delta_phase_z1
        z2 = SingleWave(grid, kx, ky, -kz, s, phase + phase_shift_z2)
        z_sum = z1 + z2

        # store frequency, period, and wave number
        self.kx = z1.kx; self.ky = z1.ky; self.kz = z1.kz
        if s == 0:
            pass
        else:
            self.omega = z1.omega
            self.period = z1.period

        self.u[:] = z_sum.u; self.v[:] = z_sum.v
        self.w[:] = z_sum.w; self.b[:] = z_sum.b

        return

# remove symbols from namespace
del Grid, State