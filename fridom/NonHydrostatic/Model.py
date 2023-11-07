import numpy as np
from IPython.display import Video
from tqdm import tqdm

from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State
from fridom.NonHydrostatic.BoundaryConditions import PBoundary
from fridom.NonHydrostatic.Source import Source
from fridom.Framework.FieldVariable import FieldVariable
from fridom.Framework.NetCDFWriter import NetCDFWriter
from fridom.Framework.ModelBase import ModelBase
from fridom.Framework.Animation import LiveAnimation, VideoAnimation


class Model(ModelBase):
    """
    A 3D non-hydrostatic Boussinesq model. Based on the ps3D model by
    Prof. Carsten Eden ( https://github.com/ceden/ps3D ).

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
        solve_pressure()        : Solve for pressure.
        adam_bashforth()        : Perform Adam-Bashforth time stepping.
        remove_pressure_gradient() : Remove pressure gradient.
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
        self.p = FieldVariable(mset, grid, 
                    name="Pressure p", bc=PBoundary(mset))
        self.div = FieldVariable(mset, grid,
                    name="Divergence", bc=PBoundary(mset))
        
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
        self.timer.add_component("Pressure Solve")
        self.timer.add_component("Pressure Gradient")
        self.timer.add_component("Live Plotting")
        self.timer.add_component("Video Writer")

        # netcdf writer
        self.writer = None
        if mset.enable_snap:
            var_names = ["u", "v", "w", "b", "p"]
            var_long_names = ["Velocity u", "Velocity v", "Velocity w", 
                              "Buoyancy b", "Pressure p"]
            var_unit_names = ["m/s", "m/s", "m/s", "m/s^2", "m^2/s^2"]
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
        self.dz1 = mset.dtype(1) / mset.dz
        self.dx2 = self.dx1**2
        self.dy2 = self.dy1**2
        self.dz2 = self.dz1**2
        self.half = mset.dtype(0.5)
        self.quarter = mset.dtype(0.25)

        # pressure solver
        if mset.pressure_solver == "CG":
            self.solve_for_pressure = self.compile_pressure_cg()
        elif mset.pressure_solver == "Spectral":
            self.solve_for_pressure = self.compile_pressure_spectral()
        else:
            raise ValueError(
                "Unknown pressure solver: {}".format(mset.pressure_solver))
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

        # solve for pressure
        start_timer("Pressure Solve")
        self.solve_pressure()
        end_timer("Pressure Solve")

        # remove pressure gradient
        start_timer("Pressure Gradient")
        self.remove_pressure_gradient()
        end_timer("Pressure Gradient")

        # write snapshot
        start_timer("Write Snapshot")
        if self.mset.enable_snap:
            if (self.it % self.mset.snap_interval) == 0:
                vars = [self.z.u, self.z.v, self.z.w, self.z.b, self.p]
                self.writer.write_cdf(vars, self.time)
        end_timer("Write Snapshot")

        # live animation
        start_timer("Live Plotting")
        if self.mset.enable_live_anim:
            if (self.it % self.mset.live_plot_interval) == 0:
                self.live_animation.update(z=self.z, p=self.p, time=self.time)
        end_timer("Live Plotting")

        # vid animation
        start_timer("Video Writer")
        if self.mset.enable_vid_anim:
            if (self.it % self.mset.vid_anim_interval) == 0:
                self.vid_animation.update(z=self.z.cpu(), p=self.p.cpu(), time=self.time)
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
        u = self.z.u; v = self.z.v; w = self.z.w; b = self.z.b
        dsqr = self.mset.dsqr
        f_cor = self.grid.f_array
        N2 = self.grid.N2_array
        cp = self.z.cp

        # Padding for averaging
        up = cp.pad(u, ((1,0), (0,1), (0,0)), 'wrap')
        vp = cp.pad(v, ((0,1), (1,0), (0,0)), 'wrap')
        wp = cp.pad(w, ((0,0), (0,0), (1,0)), 'wrap')
        bp = cp.pad(b, ((0,0), (0,0), (0,1)), 'wrap')

        # No boundary conditions required here

        # Slices
        f = slice(1,None); b = slice(None,-1)
        q = self.quarter  # 0.25
        h = self.half     # 0.5

        # calculate u-tendency
        dz.u[:] = (vp[f,f] + vp[f,b] + vp[b,f] + vp[b,b]) * q * f_cor

        # calculate v-tendency
        dz.v[:] = (up[f,f] + up[f,b] + up[b,f] + up[b,b]) * q * (-f_cor)

        # calculate w-tendency
        dz.w[:] = (bp[:,:,f] + bp[:,:,b]) * h / dsqr
        
        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1,:] = 0
        if not self.mset.periodic_bounds[2]:
            dz.w[:,:,-1] = 0

        # calculate b-tendency
        dz.b[:] = - (wp[:,:,f] + wp[:,:,b]) * h * N2

        return dz

    def nonlinear_dz(self) -> State:
        """
        Calculate nonlinear tendency.

        Returns:
            dz (State)  : Nonlinear tendency.
        """
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx1 = self.dx1; dy1 = self.dy1; dz1 = self.dz1
        quarter = self.quarter

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c,c); xb = (b,c,c)
        yf = (c,f,c); yb = (c,b,c)
        zf = (c,c,f); zb = (c,c,b)
        cc = (c,c,c)
        
        # Padding with periodic boundary conditions
        periodic = lambda x: self.z.cp.pad(x, ((2,2), (2,2), (2,2)), 'wrap')
        u  = periodic(self.z.u)
        v  = periodic(self.z.v)
        w  = periodic(self.z.w)
        bu = periodic(self.z.b)

        # boundary conditions
        if not self.mset.periodic_bounds[0]:
            u[:2,:,:]  = 0; v[:2,:,:]  = 0; w[:2,:,:]  = 0; bu[:2,:,:]  = 0
            u[-2:,:,:] = 0; v[-2:,:,:] = 0; w[-2:,:,:] = 0; bu[-2:,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            u[:,:2,:]  = 0; v[:,:2,:]  = 0; w[:,:2,:]  = 0; bu[:,:2,:]  = 0
            u[:,-2:,:] = 0; v[:,-2:,:] = 0; w[:,-2:,:] = 0; bu[:,-2:,:] = 0
        if not self.mset.periodic_bounds[2]:
            u[:,:,:2]  = 0; v[:,:,:2]  = 0; w[:,:,:2]  = 0; bu[:,:,:2]  = 0
            u[:,:,-2:] = 0; v[:,:,-2:] = 0; w[:,:,-2:] = 0; bu[:,:,-2:] = 0


        # function to calculate nonlinear tendency
        def flux_divergence(p, slice):
            """
            Calculate the flux divergence of a field p.

            Args:
                p (FieldVariable)   : Field variable.
                slice (tuple)       : Slicing tuple for averaging.

            Returns:
                res (FieldVariable) : Flux divergence.
            """
            # flux in x-,y-,z-direction
            fe = (u[cc] + u[slice]) * (p[cc] + p[xf]) * quarter
            fn = (v[cc] + v[slice]) * (p[cc] + p[yf]) * quarter
            ft = (w[cc] + w[slice]) * (p[cc] + p[zf]) * quarter

            res = (fe[cc] - fe[xb])*dx1 + \
                  (fn[cc] - fn[yb])*dy1 + \
                  (ft[cc] - ft[zb])*dz1
            return res

        # calculate nonlinear tendency
        dz.u[:] = flux_divergence(u, xf)
        dz.v[:] = flux_divergence(v, yf)
        dz.w[:] = flux_divergence(w, zf)
        dz.b[:] = flux_divergence(bu, cc)

        return dz * (-1)

    def harmonic_dz(self) -> State:
        """
        Calculate tendency due to harmonic friction / mixing.

        Returns:
            dz (State)  : Harmonic tendency.
        """
        dz = State(self.mset, self.grid)
        u = self.z.u; v = self.z.v; w = self.z.w; b = self.z.b
        dx2 = self.mset.dx**2; dy2 = self.mset.dy**2; dz2 = self.mset.dz**2
        ah = self.mset.ah; av = self.mset.av; 
        kh = self.mset.kh; kv = self.mset.kv

        # [TODO] boundary conditions
        dz.u = (u.diff_2(0) + u.diff_2(1))*ah + u.diff_2(2)*av
        dz.v = (v.diff_2(0) + v.diff_2(1))*ah + v.diff_2(2)*av
        dz.w = (w.diff_2(0) + w.diff_2(1))*ah + w.diff_2(2)*av
        dz.b = (b.diff_2(0) + b.diff_2(1))*kh + b.diff_2(2)*kv

        return dz
    
    def biharmonic_dz(self) -> State:
        """
        Calculate tendency due to biharmonic friction / mixing.

        Returns:
            dz (State)  : Biharmonic tendency.
        """
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx2 = self.dx2; dy2 = self.dy2; dz2 = self.dz2
        ahbi = self.mset.ahbi; avbi = self.mset.avbi; 
        khbi = self.mset.khbi; kvbi = self.mset.kvbi

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c,c); xb = (b,c,c)
        yf = (c,f,c); yb = (c,b,c)
        zf = (c,c,f); zb = (c,c,b)
        cc = (c,c,c)

        def biharmonic_function(p, h_coeff, v_coeff):
            """
            Calculate biharmonic friction / mixing.

            Args:
                p (FieldVariable)   : Field variable.

            Returns:
                res (FieldVariable) : Biharmonic friction / mixing.
            """
            # Padding with periodic boundary conditions
            p = self.z.cp.pad(p, ((2,2), (2,2), (2,2)), 'wrap')

            # Apply boundary conditions
            if not self.mset.periodic_bounds[0]:
                p[:2,:,:]  = 0; p[-2:,:,:] = 0
            if not self.mset.periodic_bounds[1]:
                p[:,:2,:]  = 0; p[:,-2:,:] = 0
            if not self.mset.periodic_bounds[2]:
                p[:,:,:2]  = 0; p[:,:,-2:] = 0

            # first two derivatives
            tmp_h = (p[xf] - 2*p[cc] + p[xb])*dx2*h_coeff + \
                    (p[yf] - 2*p[cc] + p[yb])*dy2*h_coeff
            tmp_v = (p[zf] - 2*p[cc] + p[zb])*dz2*v_coeff

            # last two derivatives
            res = (tmp_h[xf] - 2*tmp_h[cc] + tmp_h[xb])*dx2 + \
                  (tmp_h[yf] - 2*tmp_h[cc] + tmp_h[yb])*dy2 + \
                  (tmp_v[zf] - 2*tmp_v[cc] + tmp_v[zb])*dz2
            return res


        # biharmonic friction / mixing
        dz.u[:] = biharmonic_function(self.z.u, ahbi, avbi)
        dz.v[:] = biharmonic_function(self.z.v, ahbi, avbi)
        dz.w[:] = biharmonic_function(self.z.w, ahbi, avbi)
        dz.b[:] = biharmonic_function(self.z.b, khbi, kvbi)

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
    #   PRESSURE SOLVER
    # ============================================================

    def compile_pressure_cg(self):
        """
        Return a function that solves a divergence field for pressure 
        with a conjugate gradient solver.
        """
        if self.mset.gpu:
            from cupyx.scipy.sparse.linalg import cg, LinearOperator
        else:
            from scipy.sparse.linalg import cg, LinearOperator
        
        # shorthand notation
        Nx  = self.mset.N[0]; Ny  = self.mset.N[1]; Nz  = self.mset.N[2]
        dx2 = self.dx2;       dy2 = self.dy2;       dz2 = self.dz2
        NxNyNz = self.mset.total_grid_points
        dsqr   = self.mset.dsqr
        cp     = self.z.cp
        maxiter = self.mset.max_cg_iter
        cg_tol  = self.mset.cg_tol
        

        # define linear operator
        def laplace(p):
            p = p.reshape(Nx,Ny,Nz)
            p_pad = cp.pad(p, ((1,1),(1,1),(1,1)), "wrap")

            f = slice(2,None); b = slice(None,-2); c = slice(1,-1)
            xb = (b,c,c); xf = (f,c,c); xc = (c,c,c)
            yb = (c,b,c); yf = (c,f,c); yc = (c,c,c)
            zb = (c,c,b); zf = (c,c,f); zc = (c,c,c)

            p_laplace = ((p_pad[xf] -2*p_pad[xc] + p_pad[xb]) * dx2 +
                        (p_pad[yf] -2*p_pad[yc] + p_pad[yb]) * dy2 +
                        (p_pad[zf] -2*p_pad[zc] + p_pad[zb]) * dz2 / dsqr )
            return p_laplace.reshape(-1)

        A = LinearOperator((NxNyNz, NxNyNz), matvec=laplace)

        def solve_for_pressure(div, p0):
            p, info = cg(A, div.reshape(-1), x0=p0.reshape(-1), 
                         tol=cg_tol, maxiter=maxiter)
            return p.reshape(Nx,Ny,Nz)

        return solve_for_pressure


    def compile_pressure_spectral(self):
        """
        Return a function that solves a divergence field for pressure 
        in spectral space.
        """
        def solve_for_pressure(div, p0):
            ps = div.fft() / (-self.grid.k2_hat)
            ps[self.grid.k2_hat_zero] = 0
            return ps.fft()
        return solve_for_pressure


    def solve_pressure(self) -> None:
        """
        Calculate divergence and solve for pressure.
        """
        cp = self.z.cp
        u_pad = cp.pad(self.z.u, ((1,0), (0,0), (0,0)), 'wrap')
        v_pad = cp.pad(self.z.v, ((0,0), (1,0), (0,0)), 'wrap')
        w_pad = cp.pad(self.z.w, ((0,0), (0,0), (1,0)), 'wrap')

        # boundary conditions
        if not self.mset.periodic_bounds[0]:
            u_pad[0,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            v_pad[:,0,:] = 0
        if not self.mset.periodic_bounds[2]:
            w_pad[:,:,0] = 0

        # Slices
        c = slice(1,None); b = slice(None,-1); n = slice(None)
        xb = (b,n,n); xc = (c,n,n)
        yb = (n,b,n); yc = (n,c,n)
        zb = (n,n,b); zc = (n,n,c)

        # calculate divergence
        self.div[:] = (u_pad[xc] - u_pad[xb])*self.dx1 + \
                      (v_pad[yc] - v_pad[yb])*self.dy1 + \
                      (w_pad[zc] - w_pad[zb])*self.dz1
        self.div /= self.mset.dt

        # solve for pressure
        self.p[:] = self.solve_for_pressure(self.div, self.p)
        return


    def remove_pressure_gradient(self) -> None:
        """
        Remove pressure gradient from velocity.
        """
        cp = self.z.cp
        p_pad = cp.pad(self.p, ((0,1), (0,1), (0,1)), 'wrap')

        # Slices
        c = slice(None,-1); f = slice(1,None)
        xf = (f,c,c); xc = (c,c,c)
        yf = (c,f,c); yc = (c,c,c)
        zf = (c,c,f); zc = (c,c,c)

        # remove pressure gradient
        dt = self.mset.dt
        self.z.u -= (p_pad[xf] - p_pad[xc])*self.dx1 * dt
        self.z.v -= (p_pad[yf] - p_pad[yc])*self.dy1 * dt
        self.z.w -= (p_pad[zf] - p_pad[zc])*self.dz1 * dt / self.mset.dsqr

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            self.z.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            self.z.v[:,-1,:] = 0
        if not self.mset.periodic_bounds[2]:
            self.z.w[:,:,-1] = 0

        return


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
        self.p = self.p*0
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
            out += "hor. CFL = {:.2f},           ".format(self.z.max_cfl_h())
            out += "vert. CFL = {:.2f}".format(self.z.max_cfl_v())
            print(out)
        return

    def set_live_animation(self, live_plotter):
        if self.mset.enable_live_anim:
            self.live_animation = LiveAnimation(live_plotter)
        return

    def set_vid_animation(self, vid_plotter):
        if self.mset.enable_vid_anim:
            self.vid_animation = VideoAnimation(
                vid_plotter, self.mset.vid_anim_filename, self.mset.vid_fps)

    def show_video(self):
        if self.mset.enable_vid_anim:
            return Video(self.vid_animation.filename, width=600, embed=True) 