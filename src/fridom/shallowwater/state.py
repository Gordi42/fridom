import fridom.framework as fr
import fridom.shallowwater as sw

NEUMANN = fr.grid.BCType.NEUMANN
DIRICHLET = fr.grid.BCType.DIRICHLET

@fr.utils.jaxify
class State(fr.StateBase):
    r"""
    State vector of the 2D shallow water model.

    .. math::
        \boldsymbol{z} = \begin{pmatrix} u \\ v \\ p \end{pmatrix}

    where :math:`u` and :math:`v` are the velocity components in the 
    x- and y-directions, and :math:`p=g\eta` is the pressure field, with
    :math:`\eta` the free surface elevation.
    """
    def __init__(self, 
                 mset: 'sw.ModelSettings', 
                 is_spectral: bool = False, 
                 field_list = None) -> None:

        if field_list is None:
            cell_center = mset.grid.cell_center

            u = fr.FieldVariable(
                mset,
                name="u", 
                long_name="u - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=0),
                bc_types=(DIRICHLET, NEUMANN),
                flags=["ENABLE_FRICTION"],
                )

            v = fr.FieldVariable(
                mset,
                name="v", 
                long_name="v - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=1),
                bc_types=(NEUMANN, DIRICHLET),
                flags=["ENABLE_FRICTION"],
                )

            p = fr.FieldVariable(
                mset,
                name="p", 
                long_name="pressure", 
                units="m²/s²", 
                is_spectral=is_spectral, 
                position=cell_center,
                bc_types=(DIRICHLET, DIRICHLET),
                )

            field_list = [u, v, p]

            # add the fields from the custom field list
            for kw in mset.custom_fields:
                # Set default parameters if not provided
                if "position" not in kw:
                    # default position is cell center
                    kw["position"] = cell_center
                if "bc_types" not in kw:
                    kw["bc_types"] = (NEUMANN, NEUMANN)
                kw["mset"] = mset
                kw["is_spectral"] = is_spectral
                field_list.append(fr.FieldVariable(**kw))

        super().__init__(mset, field_list, is_spectral)
        self.__class__ = State
        return

    # ----------------------------------------------------------------
    #  State Variables
    # ----------------------------------------------------------------

    @property
    def u(self) -> fr.FieldVariable:
        """
        Velocity in the x-direction.
        """
        return self.fields["u"]

    @u.setter
    def u(self, value: fr.FieldVariable):
        self.fields["u"] = value

    @property
    def v(self) -> fr.FieldVariable:
        """
        Velocity in the y-direction.
        """
        return self.fields["v"]

    @v.setter
    def v(self, value: fr.FieldVariable):
        """
        Velocity in the y-direction.
        """
        self.fields["v"] = value

    @property
    def p(self) -> fr.FieldVariable:
        r"""
        Pressure :math:`p = g \eta`, where :math:`\eta` is the free surface elevation.
        """
        return self.fields["p"]

    @p.setter
    def p(self, value: fr.FieldVariable):
        self.fields["p"] = value

    # ----------------------------------------------------------------
    #  Energy Variables
    # ----------------------------------------------------------------

    @property
    def ekin(self) -> fr.FieldVariable:
        r"""
        Vertically integrated kinetic energy

        .. math::
            E_{\text{kin}} = \frac{Ro^2}{2} h_{\text{full}} (u^2 + v^2)

        with

        .. math::
            h_{\text{full}} = c^2 + Ro p

        Note:
            The energy is scaled with the gravity acceleration g.
        """
        if self.is_spectral:
            raise NotImplementedError("The kinetic energy is not implemented for spectral fields.")

        csqr = self.mset.csqr_field
        Ro = self.mset.Ro
        h_full = csqr + Ro * self.p
        ekin = 0.5 * Ro**2 * h_full * (self.u**2 + self.v**2)

        # Set the attributes
        ekin.name = "ekin"
        ekin.long_name = "Kinetic Energy"
        ekin.units = "m^2/s^2"
        ekin.position = self.grid.cell_center
        return ekin

    @property
    def epot(self) -> fr.FieldVariable:
        r"""
        Vertically integrated kinetic energy

        .. math::
            E_{\text{pot}} = \frac{1}{2} h_{\text{full}}^2

        with

        .. math::
            h_{\text{full}} = c^2 + Ro p

        Note:
            The energy is scaled with the gravity acceleration g.
        """
        if self.is_spectral:
            raise NotImplementedError("The kinetic energy is not implemented for spectral fields.")

        csqr = self.mset.csqr_field
        Ro = self.mset.Ro
        h_full = csqr + Ro * self.p
        epot = 0.5 * h_full ** 2

        # Set the attributes
        epot.name = "epot"
        epot.long_name = "Potential Energy"
        epot.units = "m^2/s^2"
        epot.position = self.grid.cell_center
        return epot
    
    @property
    def etot(self) -> fr.FieldVariable:
        r"""
        The total energy

        .. math::
            E_{tot} = E_{kin} + E_{pot}
        """
        etot = self.ekin + self.epot

        # Set the attributes
        etot.name = "etot"
        etot.long_name = "Total Energy"
        etot.units = "m^2/s^2"
        etot.position = self.grid.cell_center
        return etot

    # ----------------------------------------------------------------
    #  Vorticity
    # ----------------------------------------------------------------

    @property
    def rel_vort(self) -> fr.FieldVariable:
        r"""
        Relative vorticity

        .. math::
            \zeta = \partial_x v - \partial_y u
        """
        dvdx = self.v.diff(axis=0)
        dudy = self.u.diff(axis=1).interpolate(dvdx.position)
        rel_vort = dvdx - dudy

        # Set the attributes
        rel_vort.name = "rel_vort"
        rel_vort.long_name = "relative vorticity"
        rel_vort.units = "1/s"
        return rel_vort

    @property
    def pot_vort(self) -> fr.FieldVariable:
        r"""
        Scaled potential vorticity field.

        .. math::
            Q = \frac{\zeta + f \right}{c^2 + Ro p}
        
        where :math:`f` is the Coriolis parameter, and :math:`\zeta` is the 
        relative vorticity.
        """
        if self.is_spectral:
            raise NotImplementedError(
                "Potential vorticity is not implemented for spectral fields.")

        # shortcuts
        f = self.mset.f_coriolis
        csqr = self.mset.csqr_field
        Ro = self.mset.Ro

        pot_vort = (self.rel_vort + f) / (csqr + Ro * self.p)

        # Set the attributes
        pot_vort.name = "pot_vort"
        pot_vort.long_name = "Potential Vorticity"
        pot_vort.units = "s/m²"
        pot_vort.position = self.grid.cell_center

        return pot_vort

    # ----------------------------------------------------------------
    #  CFL numbers
    # ----------------------------------------------------------------

    @property
    def local_Ro(self) -> fr.FieldVariable:
        r"""
        Local Rossby number

        .. math::
            Ro_\text{local} = Ro \, \frac{\zeta_z}{f_0}

        where :math:`Ro` is the Rossby number, :math:`\zeta_z` is the vertical
        component of the relative vorticity, and :math:`f_0` is the Coriolis
        parameter.
        """
        # shortcuts
        f = self.mset.f_coriolis; Ro = self.mset.Ro

        local_Ro = Ro * self.rel_vort / f

        # Set the attributes
        local_Ro.name = "loc Ro"
        local_Ro.long_name = "Local Rossby Number"
        local_Ro.units = "1"
        
        return local_Ro

    @property
    def cfl(self) -> fr.FieldVariable:
        r"""
        The CFL number.

        .. math::
            CFL = \max  \left\{ 
                            \frac{u}{\Delta x}, \frac{v}{\Delta y} 
                        \right\} \Delta t

        where :math:`\Delta t` is the time step and :math:`\Delta x` is the
        grid spacing. 
        """
        dx, dy = self.grid.dx
        dt = self.mset.time_stepper.dt
        cfl_u = self.u.abs() * dt / dx
        cfl_v = self.v.abs() * dt / dy

        cfl = fr.config.ncp.maximum(cfl_u.arr, cfl_v.arr)

        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            arr=cfl,
            is_spectral=self.is_spectral, 
            name="cfl",
            long_name="CFL Number",
            position=self.grid.cell_center)
