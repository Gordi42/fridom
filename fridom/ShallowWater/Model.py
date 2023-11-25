from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.ShallowWater.Source import Source
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
        super().__init__(mset, grid, State)
        self.mset = mset
        
        # source term
        self.source = Source(mset, grid) if mset.enable_source else None

        # Timer
        self.timer.add_component("Linear Tendency")
        self.timer.add_component("Nonlinear Tendency")
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


        # calculate linear tendency
        start_timer("Linear Tendency")
        self.dz = self.linear_dz()
        end_timer("Linear Tendency")

        # calculate nonlinear tendency
        start_timer("Nonlinear Tendency")
        if self.mset.enable_nonlinear:
            self.dz += self.nonlinear_tendency() * self.mset.Ro
        end_timer("Nonlinear Tendency")

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

    def linear_dz(self) -> State:
        """
        Calculate linear tendency.

        Returns:
            dz (State)  : Linear tendency.
        """
        dz = State(self.mset, self.grid)
        u = self.z.u; v = self.z.v; h = self.z.h
        f_cor = self.grid.f_array
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        csqr = self.mset.csqr

        # Padding for averaging
        up = u.pad_raw(((1,1), (1,1)))
        vp = v.pad_raw(((1,1), (1,1)))
        hp = h.pad_raw(((1,1), (1,1)))

        # No boundary conditions required here

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
            dz.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1,:] = 0

        return dz

    def nonlinear_tendency(self) -> State:
        """
        Calculate nonlinear tendency.

        Returns:
            dz (State)  : Nonlinear tendency.
        """
        dz = State(self.mset, self.grid)

        # shorthand notation
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        Ro  = self.mset.Ro
        half = self.grid.half; quar = self.grid.quarter

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