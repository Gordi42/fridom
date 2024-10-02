import fridom.framework as fr
import fridom.nonhydro as nh

NEUMANN = fr.grid.BCType.NEUMANN
DIRICHLET = fr.grid.BCType.DIRICHLET

@fr.utils.jaxify
class State(fr.StateBase):
    def __init__(self, 
                 mset: 'nh.ModelSettings', 
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
                bc_types=(DIRICHLET, NEUMANN, NEUMANN),
                flags=["ENABLE_FRICTION"],
                )

            v = fr.FieldVariable(
                mset,
                name="v", 
                long_name="v - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=1),
                bc_types=(NEUMANN, DIRICHLET, NEUMANN),
                flags=["ENABLE_FRICTION"],
                )

            w = fr.FieldVariable(
                mset,
                name="w", 
                long_name="w - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=2),
                bc_types=(NEUMANN, NEUMANN, DIRICHLET),
                flags=["ENABLE_FRICTION"],
                )

            b = fr.FieldVariable(
                mset,
                name="b", 
                long_name="Buoyancy", 
                units="m/s²", 
                is_spectral=is_spectral, 
                position=cell_center,
                bc_types=(NEUMANN, NEUMANN, DIRICHLET),
                flags=["ENABLE_MIXING"],
                )

            field_list = [u, v, w, b]

            # add the fields from the custom field list
            for kw in mset.custom_fields:
                # Set default parameters if not provided
                if "position" not in kw:
                    # default position is cell center
                    kw["position"] = cell_center
                if "bc_types" not in kw:
                    kw["bc_types"] = (NEUMANN, NEUMANN, NEUMANN)
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
        return
    
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
        return
    
    @property
    def w(self) -> fr.FieldVariable:
        """
        Velocity in the z-direction.
        """
        return self.fields["w"]

    @w.setter
    def w(self, value: fr.FieldVariable):
        self.fields["w"] = value
        return
    
    @property
    def b(self) -> fr.FieldVariable:
        """
        Buoyancy
        """
        return self.fields["b"]
    
    @b.setter
    def b(self, value: fr.FieldVariable):
        self.fields["b"] = value
        return

    # ----------------------------------------------------------------
    #  Energy Variables
    # ----------------------------------------------------------------

    @property
    def ekin(self) -> fr.FieldVariable:
        r"""
        The kinetic energy

        .. math::
            E_{kin} = \frac{1}{2} (u^2 + v^2 + \delta^2 w^2)
        """
        ekin = 0.5*(self.u**2 + self.v**2 + self.mset.dsqr*self.w**2)

        # Set the attributes
        ekin.name = "ekin"
        ekin.long_name = "Kinetic Energy"
        ekin.units = "m²/s²"
        ekin.position = self.grid.cell_center
        return ekin

    @property
    def epot(self) -> fr.FieldVariable:
        r"""
        The potential energy

        If the background stratification is set, the potential energy is
        calculated as:

        .. math::
            E_{pot} = \frac{1}{2} \frac{b^2}{N^2}

        If the background stratification is not set, the potential energy is
        calculated as:

        .. math::
            E_{pot} = b z

        where :math:`z` is the vertical coordinate.
        """
        if self.mset.N2 != 0:
            epot = 0.5*(self.b**2 / self.mset.N2)
        else:
            epot = self.b * self.grid.X[2]

        # Set the attributes
        epot.name = "epot"
        epot.long_name = "Potential Energy"
        epot.units = "m²/s²"
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
        etot.units = "m²/s²"
        etot.position = self.grid.cell_center
        return etot

    # ----------------------------------------------------------------
    #  Vorticity
    # ----------------------------------------------------------------

    @property
    def rel_vort(self) -> tuple[fr.FieldVariable]:
        r"""
        The relative vorticity

        .. math::
            \boldsymbol{\zeta} = \nabla \times \boldsymbol{u}
        """
        return (self.rel_vort_x, self.rel_vort_y, self.rel_vort_z)

    @property
    def rel_vort_x(self) -> fr.FieldVariable:
        r"""
        X-component of the relative vorticity

        .. math::
            \zeta_x = \delta^2 \partial_y w - \partial_z v
        """
        dwdy = self.w.diff(axis=1)
        dvdz = self.v.diff(axis=2).interpolate(dwdy.position)
        rel_vort_x = dwdy * self.mset.dsqr - dvdz

        # Set the attributes
        rel_vort_x.name = "vort_x"
        rel_vort_x.long_name = "x component of relative vorticity"
        rel_vort_x.units = "1/s"
        return rel_vort_x

    @property
    def rel_vort_y(self) -> fr.FieldVariable:
        r"""
        Y-component of the relative vorticity

        .. math::
            \zeta_y = \partial_z u - \delta^2 \partial_x w
        """
        dudz = self.u.diff(axis=2)
        dwdx = self.w.diff(axis=0).interpolate(dudz.position)
        rel_vort_y = dudz - dwdx * self.mset.dsqr

        # Set the attributes
        rel_vort_y.name = "vort_y"
        rel_vort_y.long_name = "y component of relative vorticity"
        rel_vort_y.units = "1/s"
        return rel_vort_y

    @property
    def rel_vort_z(self) -> fr.FieldVariable:
        r"""
        Z-component of the relative vorticity (Horizontal Vorticity)

        .. math::
            \zeta_z = \partial_x v - \partial_y u
        """
        dvdx = self.v.diff(axis=0)
        dudy = self.u.diff(axis=1).interpolate(dvdx.position)
        rel_vort_z = dvdx - dudy

        # Set the attributes
        rel_vort_z.name = "vort_z"
        rel_vort_z.long_name = "horizontal vorticity"
        rel_vort_z.units = "1/s"
        return rel_vort_z

    @property
    def pot_vort(self) -> fr.FieldVariable:
        r"""
        Scaled potential vorticity field.

        .. math::
            Q = \left( f \boldsymbol{k} + Ro\,\boldsymbol{\zeta} \right) 
                \cdot \nabla \left( Ro\,b + N^2 z \right)
        
        where :math:`\boldsymbol{k}` is the vertical unit vector, :math:`f` is
        the Coriolis parameter, :math:`\boldsymbol{\zeta}` is the relative
        vorticity, :math:`b` is the buoyancy field, and :math:`N^2` is the
        buoyancy frequency.
        """
        if self.is_spectral:
            raise NotImplementedError(
                "Potential vorticity is not implemented for spectral fields.")

        # shortcuts
        f0 = self.mset.f0; N2 = self.mset.N2; Ro = self.mset.Ro

        # calculate the horizontal vorticity
        ver_vort_x = self.rel_vort_x * Ro
        ver_vort_y = self.rel_vort_y * Ro
        ver_vort_z = self.rel_vort_z * Ro

        # calculate the buoyancy gradient
        buo_grad_x, buo_grad_y, buo_grad_z = (self.b * Ro).grad()

        # interpolate the buoyancy gradient to the voriticities
        buo_grad_x = buo_grad_x.interpolate(ver_vort_x.position)
        buo_grad_y = buo_grad_y.interpolate(ver_vort_y.position)
        buo_grad_z = buo_grad_z.interpolate(ver_vort_z.position)

        # Calculate each component of the potential vorticity
        x_part = ver_vort_x * buo_grad_x
        y_part = ver_vort_y * buo_grad_y
        z_part = (ver_vort_z + f0) * (N2 + buo_grad_z)

        pot_vort = x_part + y_part + z_part

        # Set the attributes
        pot_vort.name = "pot_vort"
        pot_vort.long_name = "Potential Vorticity"
        pot_vort.units = "n/a"
        pot_vort.position = self.grid.cell_center

        return pot_vort

    @property
    def linear_pot_vort(self) -> fr.FieldVariable:
        r"""
        Linearized potential vorticity

        .. math::
            Q = Ro \left( \frac{f}{N^2} \partial_z b + \zeta_z \right)

        where :math:`Ro` is the Rossby number, :math:`f` is the Coriolis
        parameter, :math:`N^2` is the buoyancy frequency, :math:`b` is the
        buoyancy field, and :math:`\zeta_z` is the vertical component of the
        relative vorticity.
        """
        # shortcuts
        f0 = self.mset.f0; N2 = self.mset.N2; Ro = self.mset.Ro

        hor_vort = self.rel_vort_z.interpolate(self.grid.cell_center)
        dbdz = self.b.diff(axis=2).interpolate(self.grid.cell_center)
        pot_vort = Ro * (f0/N2 * dbdz + hor_vort)

        # Set the attributes
        pot_vort.name = "linear pot vort"
        pot_vort.long_name = "Linear Potential Vorticity"
        pot_vort.units = "n/a"
        pot_vort.position = self.grid.cell_center

        return pot_vort

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

        local_Ro = Ro * self.rel_vort_z / f

        # Set the attributes
        local_Ro.name = "loc Ro"
        local_Ro.long_name = "Local Rossby Number"
        local_Ro.units = "1"
        
        return local_Ro

    # ----------------------------------------------------------------
    #  CFL numbers
    # ----------------------------------------------------------------

    @property
    def cfl(self) -> fr.FieldVariable:
        r"""
        The CFL number.

        .. math::
            CFL = \max\left\{ \frac{u}{\Delta x}, \frac{v}{\Delta y}, 
                              \frac{w}{\Delta z} \right\} \Delta t

        where :math:`\Delta t` is the time step and :math:`\Delta x` is the
        grid spacing. 

        Returns:
            cfl (FieldVariable)  : Horizontal CFL number.
        """
        dx, dy, dz = self.grid.dx
        dt = self.mset.time_stepper.dt
        cfl_u = self.u.abs() * dt / dx
        cfl_v = self.v.abs() * dt / dy
        cfl_w = self.w.abs() * dt / dz

        cfl = fr.config.ncp.maximum(cfl_u.arr, cfl_v.arr)
        cfl = fr.config.ncp.maximum(cfl, cfl_w.arr)

        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            arr=cfl,
            is_spectral=self.is_spectral, 
            name="cfl",
            long_name="CFL Number",
            position=self.grid.cell_center)


@fr.utils.jaxify
class DiagnosticState(fr.StateBase):
    def __init__(self, 
                 mset: 'nh.ModelSettings', 
                 is_spectral=False, 
                 field_list=None) -> None:
        from fridom.framework.field_variable import FieldVariable
        if field_list is None:
            p = FieldVariable(
                mset, 
                name="p", 
                long_name="Pressure",
                units="m²/s",
                is_spectral=is_spectral, 
                position=mset.grid.cell_center)

            div = FieldVariable(
                mset,
                name="div", 
                long_name="Divergence",
                units="1/s",
                is_spectral=is_spectral, 
                position=mset.grid.cell_center)

            field_list = [p, div]
        super().__init__(mset, field_list, is_spectral)
        self.constructor = DiagnosticState
        return

    @property
    def p(self) -> fr.FieldVariable:
        """The pressure field."""
        return self.fields["p"]

    @p.setter
    def p(self, value: fr.FieldVariable) -> None:
        self.fields["p"] = value
    
    @property
    def div(self) -> fr.FieldVariable:
        """The divergence field."""
        return self.fields["div"]
    
    @div.setter
    def div(self, value: fr.FieldVariable) -> None:
        self.fields["div"] = value
