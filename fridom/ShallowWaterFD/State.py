from fridom.ShallowWaterFD.ModelSettings import ModelSettings
from fridom.ShallowWaterFD.Grid import Grid
from fridom.ShallowWaterFD.BoundaryConditions import UBoundary, VBoundary, HBoundary
from fridom.Framework.FieldVariable import FieldVariable
from fridom.Framework.StateBase import StateBase


class State(StateBase):
    def __init__(self, mset: ModelSettings, grid: Grid, is_spectral=False, field_list=None) -> None:
        if field_list is None:
            u = FieldVariable(mset, grid,
                name="Velocity u", is_spectral=is_spectral, bc=UBoundary(mset))
            v = FieldVariable(mset, grid,
                name="Velocity v", is_spectral=is_spectral, bc=VBoundary(mset))
            h = FieldVariable(mset, grid,
                name="Layer Thickness h", is_spectral=is_spectral, bc=HBoundary(mset))
            field_list = [u, v, h]
        super().__init__(mset, grid, field_list, is_spectral)
        self.mset = mset
        self.constructor = State
        return
    
    # ======================================================================
    #  ENERGY
    # ======================================================================

    def ekin(self) -> FieldVariable:
        """
        Calculate the kinetic energy field.
        $ ekin = 0.5 * Ro * h_full * (u^2 + v^2) $
        $ h_full = c^2 + Ro * h $

        Returns:
            ekin (FieldVariable)  : Kinetic energy field.
        """
        # First transform to physical space if necessary
        z = self
        if self.is_spectral:
            z = self.fft()

        csqr = self.mset.csqr; Ro = self.mset.Ro
        h_full = csqr + Ro * self.h
        ekin = 0.5 * Ro * h_full * (z.u**2 + z.v**2)
        return FieldVariable(self.mset, self.grid, is_spectral=False, 
                             name="Kinetic Energy", arr=ekin, 
                             bc=HBoundary(self.mset))

    def epot(self) -> FieldVariable:
        """
        Calculate the potential energy field.
        $ epot = 0.5 * h_full * h $
        $ h_full = c^2 + Ro * h $

        Returns:
            epot (FieldVariable)  : Potential energy field.
        """
        # First transform to physical space if necessary
        z = self
        if self.is_spectral:
            z = self.fft()

        csqr = self.mset.csqr; Ro = self.mset.Ro
        h_full = csqr + Ro * self.h
        epot = 0.5 * h_full * z.h

        return FieldVariable(self.mset, self.grid, is_spectral=False,
                             name="Potential Energy", arr=epot, 
                             bc=HBoundary(self.mset))
    
    def etot(self) -> FieldVariable:
        """
        Calculate the total energy field.
        $ etot = ekin + epot $

        Returns:
            etot (FieldVariable)  : Total energy field.
        """
        etot = (self.ekin() + self.epot()).arr
        return FieldVariable(self.mset, self.grid, is_spectral=False,
                             name="Total Energy", arr=etot, 
                             bc=HBoundary(self.mset))

    def mean_ekin(self) -> float:
        """
        Calculate the mean kinetic energy.

        Returns:
            mean_ekin (float)  : Mean kinetic energy.
        """
        return self.cp.mean(self.ekin())
    
    def mean_epot(self) -> float:
        """
        Calculate the mean potential energy.

        Returns:
            mean_epot (float)  : Mean potential energy.
        """
        return self.cp.mean(self.epot())
    
    def mean_etot(self) -> float:
        """
        Calculate the mean total energy.

        Returns:
            mean_etot (float)  : Mean total energy.
        """
        return self.cp.mean(self.etot())

    # ======================================================================
    #  VORTICITY
    # ======================================================================

    def hor_vort(self) -> FieldVariable:
        """
        Calculate the horizontal vorticity field.
        $ hor_vort = \partial_x v - \partial_y u $

        Returns:
            hor_vort (FieldVariable)  : Horizontal vorticity field.
        """
        # shortcuts
        dx = self.mset.dx
        dy = self.mset.dy
        u  = self.u
        v  = self.v

        # calculate the horizontal vorticity
        vort = ((v.diff_forward(0)/dx -  u.diff_forward(1)/dy)
                ).ave(-1, 0).ave(-1, 1)

        # Create the field variable
        field = FieldVariable(self.mset, self.grid, is_spectral=False, 
                              name="Horizontal Vorticity", arr=vort, bc=HBoundary(self.mset))
        return field

    def pot_vort(self) -> FieldVariable:
        """
        Calculate the potential vorticity field.
        q = (Zeta + f0) / (c² + Ro*h)

        Returns:
            pot_vort (FieldVariable)  : Scaled potential vorticity field.
        """

        # shortcuts
        f0 = self.mset.f0; 
        Ro = self.mset.Ro; csqr = self.mset.csqr

        rel_vort = self.hor_vort()
        h_full = csqr + Ro * self.h
        pot_vort = (rel_vort + f0) / h_full

        # Create the field variable
        field = FieldVariable(self.mset, self.grid, arr=pot_vort,
                              is_spectral=False, name="Potential Vorticity", bc=HBoundary(self.mset))
        return field

    # ======================================================================
    #  CFL AND PECLET NUMBERS
    # ======================================================================

    def max_cfl(self) -> float:
        """
        Calculate the maximum horizontal CFL number.

        Returns:
            max_cfl (float)  : Maximum horizontal CFL number.
        """
        u = self.u; v = self.v
        sqrt = self.cp.sqrt; maxi = self.cp.max
        dx = min(self.mset.dx, self.mset.dy)
        return sqrt(maxi(u**2 + v**2)) * self.mset.dt / dx

    def pecl(self) -> float:
        """
        Calculate the horizontal Peclet number.

        Returns:
            pecl (float)  : Horizontal Peclet number.
        """
        umax = self.cp.max(self.u**2 + self.v**2)
        res = min(umax*self.mset.dx/(1e-32 + self.mset.ah),
                  umax*self.mset.dx**3/(1e-32 + self.mset.ahbi))
        return res

    @property
    def u(self) -> FieldVariable:
        return self.field_list[0]
    
    @u.setter
    def u(self, value: FieldVariable):
        self.field_list[0] = value
        return
    
    @property
    def v(self) -> FieldVariable:
        return self.field_list[1]
    
    @v.setter
    def v(self, value: FieldVariable):
        self.field_list[1] = value
        return
    
    @property
    def h(self) -> FieldVariable:
        return self.field_list[2]

    @h.setter
    def h(self, value: FieldVariable):
        self.field_list[2] = value
        return
    
    