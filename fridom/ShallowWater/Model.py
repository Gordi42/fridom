import numpy as np
from IPython.display import Video
from tqdm import tqdm

from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.ShallowWater.Source import Source
from fridom.Framework.NetCDFWriter import NetCDFWriter
from fridom.Framework.ModelBase import ModelBase
from fridom.Framework.Animation import LiveAnimation, VideoAnimation


class Model(ModelBase):
    """
    TODO

    Attributes:
        z (State)               : State variables (u,v,w,b).
        p (FieldVariable)       : Pressure (p).
        it (int)                : Iteration counter.
        timer (TimingModule)    : Timer.
        writer (NetCDFWriter)   : NetCDF writer.

    Methods:
        step()                  : Perform one time step.
        run()                   : Run the model for a given number of steps.
        reset()                 : Reset the model (pointers, tendencies)
        diagnose()              : Print diagnostic information.
        update_pointer()        : Update pointer for Adam-Bashforth time stepping.
        update_coeff_AB()       : Update coeffs for Adam-Bashforth time stepping.
        linear_dz()             : Calculate linear tendency.
        nonlinear_dz()          : Calculate nonlinear tendency.
        harmonic_dz()           : Calculate harmonic tendency.
        biharmonic_dz()         : Calculate biharmonic tendency.
        adam_bashforth()        : Perform Adam-Bashforth time stepping.
    """

    def __init__(self, mset:ModelSettings, grid:Grid) -> None:
        """
        Constructor.

        Args:
            mset (ModelSettings)    : Model settings.
            grid (Grid)             : Grid.
        """
        super().__init__(mset, grid)
        self.mset = mset

        # Model state variables
        self.z = State(mset, grid)
        
        # source term
        self.source = Source(mset, grid) if mset.enable_source else None

        # time stepping variables
        self.dz = [State(mset, grid) for _ in range(mset.time_levels)]
        self.pointer = np.arange(mset.time_levels, dtype=np.int32)
        self.coeff_AB = np.zeros(mset.time_levels, dtype=mset.dtype)

        # Timer
        self.timer.add_component("Diagnose")
        self.timer.add_component("Write Snapshot")
        self.timer.add_component("Linear Tendency")
        self.timer.add_component("Nonlinear Tendency")
        self.timer.add_component("Harmonic Tendency")
        self.timer.add_component("Biharmonic Tendency")
        self.timer.add_component("Source Tendency")
        self.timer.add_component("Adam Bashforth Stepping")
        self.timer.add_component("Live Plotting")
        self.timer.add_component("Video Writer")

        # netcdf writer
        self.writer = None
        if mset.enable_snap:
            var_names = ["u", "v", "h"]
            var_long_names = ["Velocity u", "Velocity v", "Layer thickness h"] 
            var_unit_names = ["m/s", "m/s", "m"]
            self.writer = NetCDFWriter(mset, grid, var_names, var_long_names, 
                                       var_unit_names)

        # live animation
        self.live_animation = None
        self.set_live_animation(mset.live_plotter)

        # vid animation
        self.vid_animation = None
        self.set_vid_animation(mset.vid_plotter)

        # constants
        self.dx1 = mset.dtype(1) / mset.dx
        self.dy1 = mset.dtype(1) / mset.dy
        self.dx2 = self.dx1**2
        self.dy2 = self.dy1**2
        self.half = mset.dtype(0.5)
        self.quarter = mset.dtype(0.25)
        return


    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run(self, steps=None, runlen=None) -> None:
        """
        Run the model for a given number of steps or a given time.

        Args:
            steps (int)     : Number of steps to run.
            runlen (float)  : Time to run. (preferred over steps)
        """
        # check if steps or runlen is given
        if runlen is not None:
            steps = runlen / self.mset.dt
        
        # progress bar
        tq = tqdm if self.mset.enable_tqdm else lambda x: x

        # start vid animation
        if self.mset.enable_vid_anim:
            self.vid_animation.start_writer()
        
        # main loop
        self.timer.total.start()
        for _ in tq(range(int(steps))):
            self.step()
        self.timer.total.stop()

        # close netcdf writer
        if self.mset.enable_snap:
            self.writer.close()

        # stop vid animation
        if self.mset.enable_vid_anim:
            self.vid_animation.stop_writer()

        return

    
    # ============================================================
    #   SINGLE TIME STEP
    # ============================================================

    def step(self) -> None:
        """
        Update the model state by one time step.
        """
        self.update_pointer()
        self.update_coeff_AB()
        
        start_timer = lambda x: self.timer.get(x).start()
        end_timer   = lambda x: self.timer.get(x).stop()

        # diagnose
        start_timer("Diagnose")
        self.diagnose()
        end_timer("Diagnose")

        # calculate linear tendency
        start_timer("Linear Tendency")
        self.dz[self.pointer[0]] = self.linear_dz()
        end_timer("Linear Tendency")

        # calculate nonlinear tendency
        start_timer("Nonlinear Tendency")
        if self.mset.enable_nonlinear:
            self.dz[self.pointer[0]] += self.nonlinear_dz() * self.mset.Ro
        end_timer("Nonlinear Tendency")

        # calculate harmonic tendency
        start_timer("Harmonic Tendency")
        if self.mset.enable_harmonic:
            self.dz[self.pointer[0]] += self.harmonic_dz()
        end_timer("Harmonic Tendency")

        # calculate biharmonic tendency
        start_timer("Biharmonic Tendency")
        if self.mset.enable_biharmonic:
            self.dz[self.pointer[0]] -= self.biharmonic_dz()
        end_timer("Biharmonic Tendency")

        # calculate source tendency
        start_timer("Source Tendency")
        if self.mset.enable_source:
            self.dz[self.pointer[0]] += self.source_dz()
        end_timer("Source Tendency")

        # Adam Bashforth time stepping
        start_timer("Adam Bashforth Stepping")
        self.adam_bashforth()
        end_timer("Adam Bashforth Stepping")

        # write snapshot
        start_timer("Write Snapshot")
        if self.mset.enable_snap:
            if (self.it % self.mset.snap_interval) == 0:
                vars = [self.z.u, self.z.v, self.z.h]
                self.writer.write_cdf(vars, self.time)
        end_timer("Write Snapshot")

        # live animation
        start_timer("Live Plotting")
        if self.mset.enable_live_anim:
            if (self.it % self.mset.live_plot_interval) == 0:
                self.live_animation.update(z=self.z, time=self.time)
        end_timer("Live Plotting")

        # vid animation
        start_timer("Video Writer")
        if self.mset.enable_vid_anim:
            if (self.it % self.mset.vid_anim_interval) == 0:
                self.vid_animation.update(z=self.z.cpu(), time=self.time)
        end_timer("Video Writer")

        self.it += 1
        return


    # ============================================================
    #   TENDENCIES
    # ============================================================

    def linear_dz(self) -> State:
        """
        Calculate linear tendency.

        Returns:
            dz (State)  : Linear tendency.
        """
        dz = State(self.mset, self.grid)
        u = self.z.u; v = self.z.v; h = self.z.h
        f_cor = self.grid.f_array
        cp = self.z.cp
        dx1 = self.dx1; dy1 = self.dy1
        csqr = self.mset.csqr

        # Padding for averaging
        up = u.pad_raw(((1,1), (1,1)))
        vp = v.pad_raw(((1,1), (1,1)))
        hp = h.pad_raw(((1,1), (1,1)))

        # No boundary conditions required here

        # Slices
        f = slice(2,None); b = slice(None,-2); c = slice(1,-1)
        q = self.quarter  # 0.25

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
            dz.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1,:] = 0

        return dz

    def nonlinear_dz(self) -> State:
        """
        Calculate nonlinear tendency.

        Returns:
            dz (State)  : Nonlinear tendency.
        """
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx1 = self.dx1; dy1 = self.dy1
        Ro  = self.mset.Ro
        half = self.half; quar = self.quarter

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c); xb = (b,c); xc = (c,c)
        yf = (c,f); yb = (c,b); yc = (c,c)

        # Padding
        up = self.z.u.pad_raw(((2,2), (2,2)))
        vp = self.z.v.pad_raw(((2,2), (2,2)))
        hp = self.z.h.pad_raw(((2,2), (2,2)))

        # Apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            up[:2,:] = 0; up[-2:,:] = 0
            vp[:2,:] = 0; vp[-2:,:] = 0
            hp[:2,:] = 0; hp[-2:,:] = 0
        if not self.mset.periodic_bounds[1]:
            up[:,:2] = 0; up[:,-2:] = 0
            vp[:,:2] = 0; vp[:,-2:] = 0
            hp[:,:2] = 0; hp[:,-2:] = 0

        # advection of layer thickness with flux divergence
        fe = up[xc] * (hp[xf] + hp[xc]) * half
        fn = vp[yc] * (hp[yf] + hp[yc]) * half

        dz.h[:] = -(fe[xc] - fe[xb])*dx1 - (fn[yc] - fn[yb])*dy1

        # advection of momentum with potential vorticity and kinetic energy
        hf = Ro * hp + self.mset.csqr
        q = (vp[xf] - vp[xc])*dx1 - (up[yf] - up[yc])*dy1
        q /= (hf[c,c] + hf[f,c] + hf[c,f] + hf[f,f])*quar
        k = (up[xc]**2 + up[xb]**2 + vp[yc]**2 + vp[yb]**2)*quar

        fe = up[xc] * (hf[xf] + hf[xc]) * half
        fn = vp[yc] * (hf[yf] + hf[yc]) * half

        dz.u[:] = (q[yc] * (fn[xc] + fn[xf]) + q[yb] * (fn[c,b] + fn[f,b]))*quar  
        dz.u[:] -= (k[xf] - k[xc])*dx1 

        dz.v[:] = -(q[xc] * (fe[yc] + fe[yf]) + q[xb] * (fe[b,c] + fe[b,f]))*quar
        dz.v[:] -= (k[yf] - k[yc])*dy1
        return dz


    def harmonic_dz(self) -> State:
        """
        Calculate tendency due to harmonic friction / mixing.

        Returns:
            dz (State)  : Harmonic tendency.
        """
        dz = State(self.mset, self.grid)
        u = self.z.u; v = self.z.v; h = self.z.h
        ah = self.mset.ah
        kh = self.mset.kh

        # [TODO] boundary conditions
        dz.u = (u.diff_2(0) + u.diff_2(1))*ah 
        dz.v = (v.diff_2(0) + v.diff_2(1))*ah 
        dz.h = (h.diff_2(0) + h.diff_2(1))*kh 

        return dz
    
    def biharmonic_dz(self) -> State:
        """
        Calculate tendency due to biharmonic friction / mixing.

        Returns:
            dz (State)  : Biharmonic tendency.
        """
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx2 = self.dx2; dy2 = self.dy2
        ahbi = self.mset.ahbi
        khbi = self.mset.khbi

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c); xb = (b,c)
        yf = (c,f); yb = (c,b)
        cc = (c,c)

        def biharmonic_function(p, coeff):
            """
            Calculate biharmonic friction / mixing.

            Args:
                p (FieldVariable)   : Field variable.

            Returns:
                res (FieldVariable) : Biharmonic friction / mixing.
            """
            # Padding with periodic boundary conditions
            p = self.z.cp.pad(p, ((2,2), (2,2)), 'wrap')

            # Apply boundary conditions
            if not self.mset.periodic_bounds[0]:
                p[:2,:]  = 0; p[-2:,:] = 0
            if not self.mset.periodic_bounds[1]:
                p[:,:2]  = 0; p[:,-2:] = 0

            # first two derivatives
            tmp_h = (p[xf] - 2*p[cc] + p[xb])*dx2*coeff + \
                    (p[yf] - 2*p[cc] + p[yb])*dy2*coeff

            # last two derivatives
            res = (tmp_h[xf] - 2*tmp_h[cc] + tmp_h[xb])*dx2 + \
                  (tmp_h[yf] - 2*tmp_h[cc] + tmp_h[yb])*dy2
            return res


        # biharmonic friction / mixing
        dz.u[:] = biharmonic_function(self.z.u, ahbi)
        dz.v[:] = biharmonic_function(self.z.v, ahbi)
        dz.h[:] = biharmonic_function(self.z.h, khbi)

        return dz

    def source_dz(self) -> State:
        """
        Calculate tendency due to source term.

        Returns:
            dz (State)  : Source tendency.
        """
        self.source.update(self.time)
        return self.source


    # ============================================================
    #   TIME STEPPING
    # ============================================================

    def adam_bashforth(self) -> None:
        """
        Perform Adam-Bashforth time stepping.
        """
        for i in range(self.mset.time_levels):
            self.z += self.dz[self.pointer[i]] * self.mset.dt * self.coeff_AB[i]
        return


    def update_pointer(self) -> None:
        """
        Update pointer for Adam-Bashforth time stepping.
        """
        self.pointer = np.roll(self.pointer, 1)
        return


    def update_coeff_AB(self) -> None:
        """
        Upward ramping of Adam-Bashforth coefficients after restart.
        """
        # current time level (ctl)
        # maximum ctl is the number of time levels - 1
        ctl = min(self.it, self.mset.time_levels-1)

        # list of Adam-Bashforth coefficients
        coeffs = [self.mset.AB1, self.mset.AB2, self.mset.AB3, self.mset.AB4]

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB[:]      = 0
        self.coeff_AB[:ctl+1] = coeffs[ctl]
        return

    # ============================================================
    #   OTHER METHODS
    # ============================================================

    def reset(self) -> None:
        """
        Reset the model (pointers, tendencies).
        """
        super().reset()
        self.z = self.z*0
        self.dz = [dz*0 for dz in self.dz]
        return


    def diagnose(self) -> None:
        """
        Print diagnostic information.
        """
        if not self.mset.enable_diag:
            return
        if (self.it % self.mset.diag_interval) == 0:
            out = "Diagnostic at t = {:.2f}\n".format(self.it * self.mset.dt)
            out += "MKE = {:.2e},    ".format(self.z.mean_ekin())
            out += "MPE = {:.2e},    ".format(self.z.mean_epot())
            out += "MTE = {:.2e}\n".format(self.z.mean_etot())
            out += "CFL = {:.2f}".format(self.z.max_cfl())
            print(out)
        return

    def set_live_animation(self, live_plotter):
        if self.mset.enable_live_anim:
            self.live_animation = LiveAnimation(live_plotter)
        return

    def set_vid_animation(self, vid_plotter):
        if self.mset.enable_vid_anim:
            self.vid_animation = VideoAnimation(
                vid_plotter, self.mset.vid_anim_filename, self.mset.vid_fps,
                self.mset.vid_max_jobs)

    def show_video(self):
        if self.mset.enable_vid_anim:
            return Video(self.vid_animation.filename, width=600, embed=True) 