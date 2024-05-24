from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.Framework.TimingModule import TimingModule

class NonlinearTendency:
    def __init__(self, mset: ModelSettings, grid: Grid, timer: TimingModule):
        """
        Constructor of the NonlinearTendency class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid
        self.timer = timer

        if mset.solver == "FD":
            self.NonlinTend = NonlinearTendencyFD(mset, grid)
        elif mset.solver == "Spectral":
            self.NonlinTend = NonlinearTendencySpectral(mset, grid)

        # add a timer for the linear tendency
        self.timer.add_component('Nonlinear Tendency')

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the nonlinear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : tendency of the state.
        """
        # start the timer
        self.timer.get("Nonlinear Tendency").start()

        self.NonlinTend(z, dz)

        # stop the timer
        self.timer.get("Nonlinear Tendency").stop()
        return


class NonlinearTendencyFD:
    def __init__(self, mset: ModelSettings, grid: Grid):
        """
        Solver for the nonlinear tendency with finite differences.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the nonlinear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Tendency of the state.
        """
        # shorthand notation
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        Ro  = self.mset.Ro
        half = self.grid.half; quar = self.grid.quarter

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c); xb = (b,c); xc = (c,c)
        yf = (c,f); yb = (c,b); yc = (c,c)

        # Padding
        up = z.u.pad_raw(((2,2), (2,2)))
        vp = z.v.pad_raw(((2,2), (2,2)))
        hp = z.h.pad_raw(((2,2), (2,2)))

        # Apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            up[:2,:] = 0; up[-3:,:] = 0
            vp[:2,:] = 0; vp[-2:,:] = 0
            hp[:2,:] = 0; hp[-2:,:] = 0
        if not self.mset.periodic_bounds[1]:
            up[:,:2] = 0; up[:,-2:] = 0
            vp[:,:2] = 0; vp[:,-3:] = 0
            hp[:,:2] = 0; hp[:,-2:] = 0

        # advection of layer thickness with flux divergence
        fe = up[xc] * (hp[xf] + hp[xc]) * half
        fn = vp[yc] * (hp[yf] + hp[yc]) * half

        dz.h[:] -= (fe[xc] - fe[xb])*dx1*Ro + (fn[yc] - fn[yb])*dy1*Ro

        # advection of momentum with potential vorticity and kinetic energy
        hf = Ro * hp + self.mset.csqr
        q = (vp[xf] - vp[xc])*dx1 - (up[yf] - up[yc])*dy1
        # apply boundary conditions to potential vorticity
        if not self.mset.periodic_bounds[0]:
            q[0,:] = -(up[2,c] - up[1,c])*dx1;
            q[-1,:] = -(up[-1,c] - up[-2,c])*dx1
        if not self.mset.periodic_bounds[1]:
            q[:,0] = (vp[c,2] - vp[c,1])*dy1;
            q[:,-1] = (vp[c,-1] - vp[c,-2])*dy1
        q /= (hf[c,c] + hf[f,c] + hf[c,f] + hf[f,f])*quar
        k = (up[xc]**2 + up[xb]**2 + vp[yc]**2 + vp[yb]**2)*quar

        fe = up[xc] * (hf[xf] + hf[xc]) * half
        fn = vp[yc] * (hf[yf] + hf[yc]) * half

        dz.u[:] += (q[yc] * (fn[xc] + fn[xf]) + q[yb] * (fn[c,b] + fn[f,b]))*quar*Ro
        dz.u[:] -= (k[xf] - k[xc])*dx1 * Ro

        dz.v[:] -= (q[xc] * (fe[yc] + fe[yf]) + q[xb] * (fe[b,c] + fe[b,f]))*quar*Ro
        dz.v[:] -= (k[yf] - k[yc])*dy1 * Ro

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1] = 0

        return


class NonlinearTendencySpectral:
    def __init__(self, mset: ModelSettings, grid: Grid):
        """
        Solver for the nonlinear tendency in spectral space.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the nonlinear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Tendency of the state.
        """
        Kx, Ky = self.grid.K

        u_hat = z.u * self.grid.dealias_mask
        v_hat = z.v * self.grid.dealias_mask
        h_hat = z.h * self.grid.dealias_mask

        u = u_hat.fft()
        v = v_hat.fft()
        h = h_hat.fft()

        zeta = (1j * Kx * v_hat - 1j * Ky * u_hat).fft()
        ekin_hat = ((u**2 + v**2)*0.5).fft()

        duhdx = 1j * Kx * (u * h).fft()
        dvhdy = 1j * Ky * (v * h).fft()
        dz.h[:] -= (duhdx + dvhdy) * self.mset.Ro

        dz.u[:] += ((zeta * v).fft() - 1j * Kx * ekin_hat ) * self.mset.Ro
        dz.v[:] -= ((zeta * u).fft() + 1j * Ky * ekin_hat ) * self.mset.Ro

        return

# remove symbols from namespace
del ModelSettings, Grid, State, TimingModule