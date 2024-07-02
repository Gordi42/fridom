# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.state_base import StateBase
from fridom.framework.field_variable import FieldVariable
from mpi4py import MPI
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.model_settings import ModelSettings

class State(StateBase):
    def __init__(self, mset: 'ModelSettings', is_spectral=False, field_list=None) -> None:
        from fridom.framework.field_variable import FieldVariable
        if field_list is None:
            u = FieldVariable(mset,
                name="Velocity u", is_spectral=is_spectral)
            v = FieldVariable(mset,
                name="Velocity v", is_spectral=is_spectral)
            w = FieldVariable(mset,
                name="Velocity w", is_spectral=is_spectral)
            b = FieldVariable(mset,
                name="Buoyancy b", is_spectral=is_spectral)
            field_list = [u, v, w, b]
        super().__init__(mset, field_list, is_spectral)
        self.constructor = State
        return
    
    # ======================================================================
    #  ENERGY
    # ======================================================================

    def ekin(self) -> FieldVariable:
        """
        Calculate the kinetic energy field.
        $ ekin = 0.5 * (u^2 + v^2 + \\delta^2 w^2) $

        Returns:
            ekin (FieldVariable)  : Kinetic energy field.
        """
        from fridom.framework.field_variable import FieldVariable
        # First transform to physical space if necessary
        z = self
        if self.is_spectral:
            z = self.fft()
        dsqr = self.mset.dsqr
        ekin = 0.5*(z.u**2 + z.v**2 + dsqr*z.w**2)
        return FieldVariable(self.mset, is_spectral=False, name="Kinetic Energy", arr=ekin)

    def epot(self) -> FieldVariable:
        """
        Calculate the potential energy field.
        $ epot = 0.5 * (b^2 / N0^2) $

        Returns:
            epot (FieldVariable)  : Potential energy field.
        """
        from fridom.framework.field_variable import FieldVariable
        # First transform to physical space if necessary
        z = self
        if self.is_spectral:
            z = self.fft()
        N2 = self.mset.N2
        epot = 0.5*(z.b**2 / (N2))
        return FieldVariable(self.mset, is_spectral=False,
                             name="Potential Energy", arr=epot)
    
    def etot(self) -> FieldVariable:
        """
        Calculate the total energy field.
        $ etot = ekin + epot $

        Returns:
            etot (FieldVariable)  : Total energy field.
        """
        from fridom.framework.field_variable import FieldVariable
        etot = (self.ekin() + self.epot()).arr
        return FieldVariable(self.mset, is_spectral=False,
                             name="Total Energy", arr=etot)

    def mean_ekin(self) -> float:
        """
        Calculate the mean kinetic energy.

        Returns:
            mean_ekin (float)  : Mean kinetic energy.
        """
        local_ekin = self.ekin() * self.grid.dV
        local_tot_ekin = config.ncp.sum(local_ekin[self.grid.inner_slice])
        global_tot_ekin = MPI.COMM_WORLD.allreduce(local_tot_ekin, op=MPI.SUM)
        return global_tot_ekin
    
    def mean_epot(self) -> float:
        """
        Calculate the mean potential energy.

        Returns:
            mean_epot (float)  : Mean potential energy.
        """
        local_epot = self.epot() * self.grid.dV
        local_tot_epot = config.ncp.sum(local_epot[self.grid.inner_slice])
        global_tot_epot = MPI.COMM_WORLD.allreduce(local_tot_epot, op=MPI.SUM)
        return global_tot_epot

    
    def mean_etot(self) -> float:
        """
        Calculate the mean total energy.

        Returns:
            mean_etot (float)  : Mean total energy.
        """
        local_etot = self.etot() * self.grid.dV
        local_tot_etot = config.ncp.sum(local_etot[self.grid.inner_slice])
        global_tot_etot = MPI.COMM_WORLD.allreduce(local_tot_etot, op=MPI.SUM)
        return global_tot_etot

    # ======================================================================
    #  VORTICITY
    # ======================================================================

    def hor_vort(self) -> FieldVariable:
        """
        Calculate the horizontal vorticity field.
        $ hor_vort = \\partial_x v - \\partial_y u $

        Returns:
            hor_vort (FieldVariable)  : Horizontal vorticity field.
        """
        from fridom.framework.field_variable import FieldVariable
        # shortcuts
        dx = self.mset.dx
        dy = self.mset.dy
        u  = self.u
        v  = self.v

        # calculate the horizontal vorticity
        vort = ((v.diff_forward(0) -  u.diff_forward(1))
                ).ave(-1, 0).ave(-1, 1)

        # Create the field variable
        field = FieldVariable(self.mset, is_spectral=False, 
                              name="Horizontal Vorticity", arr=vort)
        return field

    def ver_vort_x(self) -> FieldVariable:
        """
        Calculate the scaled vertical vorticity field in y,z-plane.
        $ ver_vort_x = \\delta^2 \\partial_y w - \\partial_z v $

        Returns:
            ver_vort_x (FieldVariable)  : Vertical vorticity field in y,z-plane.
        """
        from fridom.framework.field_variable import FieldVariable
        # shortcuts
        dy   = self.mset.dy
        dz   = self.mset.dz
        dsqr = self.mset.dsqr
        w    = self.w
        v    = self.v

        # calculate the vertical vorticity
        vort = ((w.diff_forward(1)*dsqr -  v.diff_forward(2))
                ).ave(-1, 1).ave(-1, 2)
            
        # Create the field variable
        field = FieldVariable(self.mset, is_spectral=False, 
                              name="y,z - Vorticity", arr=vort)
        return field
    
    def ver_vort_y(self) -> FieldVariable:
        """
        Calculate the scaled vertical vorticity field in x,z-plane.
        $ ver_vort_y = \\partial_z u - \\delta^2 \\partial_x w $

        Returns:
            ver_vort_y (FieldVariable)  : Vertical vorticity field in x,z-plane.
        """
        from fridom.framework.field_variable import FieldVariable
        # shortcuts
        dx   = self.mset.dx 
        dz   = self.mset.dz
        dsqr = self.mset.dsqr
        u    = self.u        
        w    = self.w

        # calculate the vertical vorticity
        vort = ((u.diff_forward(2) - dsqr * w.diff_forward(0))
                ).ave(-1, 0).ave(-1, 2)

        # Create the field variable
        field = FieldVariable(self.mset, is_spectral=False, 
                              name="x,z - Vorticity", arr=vort)
        return field

    def pot_vort(self) -> FieldVariable:
        """
        Calculate the scaled potential vorticity field.
        $ Q = (f\\vec{k} + \\nabla \\times \\vec{u}) / \\cdot \\nabla (b + N^2 z) $

        Returns:
            pot_vort (FieldVariable)  : Scaled potential vorticity field.
        """
        from fridom.framework.field_variable import FieldVariable
        # shortcuts
        f0 = self.mset.f0; N2 = self.mset.N2; 
        dx = self.mset.dx; dy = self.mset.dy; dz = self.mset.dz
        Ro = self.mset.Ro; dsqr = self.mset.dsqr
        b = self.b

        # calculate the horizontal vorticity
        ver_vort_x = self.ver_vort_x() * Ro
        ver_vort_y = self.ver_vort_y() * Ro
        hor_vort   = self.hor_vort()   * Ro

        # calculate the buoyancy gradient
        buo_grad_x = b.diff_forward(0).ave(-1,0) * Ro / (N2)
        buo_grad_y = b.diff_forward(1).ave(-1,1) * Ro / (N2)
        buo_grad_z = b.diff_forward(2).ave(-1,2) * Ro / (N2)

        # calculate the potential vorticity
        pot_vort =  (f0 + hor_vort) * (1 + buo_grad_z) + \
                    buo_grad_x * ver_vort_x + buo_grad_y * ver_vort_y

        # Create the field variable
        field = FieldVariable(self.mset, arr=pot_vort,
                              is_spectral=False, name="Potential Vorticity")
        return field

    def linear_pot_vort(self) -> FieldVariable:
        """
        Calculate the linearized scaled potential vorticity field.
        $ Q = Ro * (f/N0^2 \\partial_z b + \\zeta)$
        """
        from fridom.framework.field_variable import FieldVariable
        # shortcuts
        f0 = self.mset.f0; N2 = self.mset.N2;
        Ro = self.mset.Ro; dsqr = self.mset.dsqr
        b = self.b

        # calculate the horizontal vorticity
        hor_vort = self.hor_vort()

        # calculate the buoyancy gradient
        buo_grad_z = b.diff_forward(2).ave(-1,2)

        # calculate the potential vorticity
        pot_vort = Ro * (f0/N2 * buo_grad_z + hor_vort)

        # Create the field variable
        field = FieldVariable(self.mset, arr=pot_vort,
                              is_spectral=False, name="Linear PV")
        return field

    # ======================================================================
    #  CFL AND PECLET NUMBERS
    # ======================================================================

    def max_cfl_h(self) -> float:
        """
        Calculate the maximum horizontal CFL number.

        Returns:
            max_cfl_h (float)  : Maximum horizontal CFL number.
        """
        u = self.u; v = self.v
        sqrt = self.cp.sqrt; maxi = self.cp.max
        dx = min(self.mset.dx, self.mset.dy)
        return sqrt(maxi(u**2 + v**2)) * self.mset.dt / dx

    def max_cfl_v(self) -> float:
        """
        Calculate the maximum vertical CFL number.

        Returns:
            max_cfl_v (float)  : Maximum vertical CFL number.
        """
        w = self.w
        sqrt = self.cp.sqrt; maxi = self.cp.max
        return sqrt(maxi(w**2)) * self.mset.dt / self.mset.dz

    def pecl_h(self) -> float:
        """
        Calculate the horizontal Peclet number.

        Returns:
            pecl_h (float)  : Horizontal Peclet number.
        """
        umax = self.cp.max(self.u**2 + self.v**2 + self.w**2)
        res = min(umax*self.mset.dx/(1e-32 + self.mset.ah),
                  umax*self.mset.dx**3/(1e-32 + self.mset.ahbi))
        return res
    
    def pecl_v(self) -> float:
        """
        Calculate the vertical Peclet number.

        Returns:
            pecl_v (float)  : Vertical Peclet number.
        """
        umax = self.cp.max(self.u**2 + self.v**2 + self.w**2)
        res = min(umax*self.mset.dz/(1e-32 + self.mset.av),
                    umax*self.mset.dz**3/(1e-32 + self.mset.avbi))
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
    def w(self) -> FieldVariable:
        return self.field_list[2]

    @w.setter
    def w(self, value: FieldVariable):
        self.field_list[2] = value
        return
    
    @property
    def b(self) -> FieldVariable:
        return self.field_list[3]
    
    @b.setter
    def b(self, value: FieldVariable):
        self.field_list[3] = value
        return