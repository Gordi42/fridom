from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.Framework.TimingModule import TimingModule

class LinearTendency:
    def __init__(self, mset: ModelSettings, grid: Grid, timer: TimingModule):
        """
        Constructor of the LinearTendency class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid
        self.timer = timer

        if mset.solver == "FD":
            self.LinTend = LinearTendencyFD(mset, grid)
        elif mset.solver == "Spectral":
            self.LinTend = LinearTendencySpectral(mset, grid)

        # add a timer for the linear tendency
        self.timer.add_component('Linear Tendency')

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Linear tendency of the state.
        """
        # start the timer
        self.timer.get("Linear Tendency").start()

        self.LinTend(z, dz)

        # stop the timer
        self.timer.get("Linear Tendency").stop()
        return


class LinearTendencyFD:
    def __init__(self, mset: ModelSettings, grid: Grid):
        """
        Solver for the linear tendency with finite differences.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Linear tendency of the state.
        """
        u = z.u; v = z.v; h = z.h
        f_cor = self.grid.f_array
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        csqr = self.mset.csqr

        # Padding for averaging
        up = u.pad_raw(((1,1), (1,1)))
        vp = v.pad_raw(((1,1), (1,1)))
        hp = h.pad_raw(((1,1), (1,1)))

        if not self.mset.periodic_bounds[0]:
            up[0,:] = 0; up[-2:,:] = 0
            vp[0,:] = 0; vp[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            vp[:,0] = 0; vp[:,-2:] = 0
            up[:,0] = 0; up[:,-1] = 0

        # Slices
        f = slice(2,None); b = slice(None,-2); c = slice(1,-1)
        q = self.grid.quarter  # 0.25

        # Coriolis tendency
        dz.u[:] = (vp[c,c] + vp[f,c] + vp[c,b] + vp[f,b]) * q * f_cor
        dz.v[:] = (up[c,c] + up[b,c] + up[c,f] + up[b,f]) * q * (-f_cor)

        # Pressure gradient tendency
        dz.u[:] -= (hp[f,c] - hp[c,c]) * dx1
        dz.v[:] -= (hp[c,f] - hp[c,c]) * dy1

        # Horizontal divergence tendency
        dz.h[:] = -((up[c,c] - up[b,c])*dx1 + (vp[c,c] - vp[c,b])*dy1)*csqr

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1] = 0

        return


class LinearTendencySpectral:
    def __init__(self, mset: ModelSettings, grid: Grid):
        """
        Solver for the linear tendency in spectral space.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Linear tendency of the state.
        """
        Kx, Ky = self.grid.K

        # Coriolis tendency
        dz.u[:] = self.mset.f0 * z.v
        dz.v[:] = -self.mset.f0 * z.u

        # Pressure gradient tendency
        dz.u[:] -= 1j * Kx * z.h
        dz.v[:] -= 1j * Ky * z.h

        # Horizontal divergence tendency
        dz.h[:] = -1j * (Kx * z.u + Ky * z.v) * self.mset.csqr
        return

# remove symbols from namespace
del ModelSettings, Grid, State, TimingModule