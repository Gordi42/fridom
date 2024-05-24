from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.Framework.ModelBase import ModelBase


class Model(ModelBase):
    """
    A scaled rotating shallow water model. The discretization is based on a
    energy conserving finite difference scheme on a staggered Arakawa C-grid.
    Based on Sadourny [1975].
    """

    def __init__(self, mset:ModelSettings, grid:Grid) -> None:
        """
        Constructor.

        Args:
            mset (ModelSettings)    : Model settings.
            grid (Grid)             : Grid.
        """
        if mset.solver == "FD":
            is_spectral = False
        elif mset.solver == "Spectral":
            is_spectral = True
        from fridom.ShallowWater.State import State
        super().__init__(mset, grid, State, is_spectral=is_spectral)
        self.mset = mset

        # Modules
        from fridom.ShallowWater.Modules import LinearTendency, NonlinearTendency
        self.linear_tendency = LinearTendency(mset, grid, self.timer)
        self.nonlinear_tendency = NonlinearTendency(mset, grid, self.timer)
        
        # source term
        from fridom.ShallowWater.Source import Source
        self.source = Source(mset, grid) if mset.enable_source else None

        # Timer
        self.timer.add_component("Harmonic Tendency")
        self.timer.add_component("Biharmonic Tendency")
        self.timer.add_component("Source Tendency")

        # netcdf writer
        var_names = ["u", "v", "h"]
        var_long_names = ["Velocity u", "Velocity v", "Layer thickness h"] 
        var_unit_names = ["m/s", "m/s", "m"]
        self.writer.set_var_names(var_names, var_long_names, var_unit_names)

        return


    # ============================================================
    #   TOTAL TENDENCY
    # ============================================================

    def total_tendency(self) -> None:
        """
        Calculate total tendency. (Righthand side of the PDE)
        """
        
        start_timer = lambda x: self.timer.get(x).start()
        end_timer   = lambda x: self.timer.get(x).stop()

        self.linear_tendency(self.z, self.dz)
        if self.mset.enable_nonlinear:
            self.nonlinear_tendency(self.z, self.dz)

        # calculate harmonic tendency
        start_timer("Harmonic Tendency")
        if self.mset.enable_harmonic:
            self.dz += self.harmonic_dz()
        end_timer("Harmonic Tendency")

        # calculate biharmonic tendency
        start_timer("Biharmonic Tendency")
        if self.mset.enable_biharmonic:
            self.dz -= self.biharmonic_dz()
        end_timer("Biharmonic Tendency")

        # calculate source tendency
        start_timer("Source Tendency")
        if self.mset.enable_source:
            self.dz += self.source_dz()
        end_timer("Source Tendency")
        return

    # ============================================================
    #   TENDENCIES
    # ============================================================


    def harmonic_dz(self) -> State:
        """
        Calculate tendency due to harmonic friction / mixing.

        Returns:
            dz (State)  : Harmonic tendency.
        """
        from fridom.ShallowWater.State import State
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
        from fridom.ShallowWater.State import State
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx2 = self.grid.dx2; dy2 = self.grid.dy2
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
    #   OTHER METHODS
    # ============================================================

    def diagnostics(self) -> None:
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

    def get_writer_variables(self):
        return [self.z.u, self.z.v, self.z.h]

    def update_live_animation(self):
        self.live_animation.update(z=self.z, time=self.time)

    def update_vid_animation(self):
        self.vid_animation.update(z=self.z.cpu(), time=self.time)

# remove symbols from namespace
del ModelSettings, Grid, State, ModelBase