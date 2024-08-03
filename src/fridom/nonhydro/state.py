import fridom.framework as fr
import fridom.nonhydro as nh

class State(fr.StateBase):
    def __init__(self, 
                 mset: nh.ModelSettings, 
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
                position=cell_center.shift(axis=0, direction="forward"),
                transform_types=(fr.grid.TransformType.DST1,
                                 fr.grid.TransformType.DCT2,
                                 fr.grid.TransformType.DCT2)
                )

            v = fr.FieldVariable(
                mset,
                name="v", 
                long_name="v - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=1, direction="forward"),
                transform_types=(fr.grid.TransformType.DCT2,
                                 fr.grid.TransformType.DST1,
                                 fr.grid.TransformType.DCT2)
                )

            w = fr.FieldVariable(
                mset,
                name="w", 
                long_name="w - velocity",
                units="m/s", 
                is_spectral=is_spectral, 
                position=cell_center.shift(axis=2, direction="forward"),
                transform_types=(fr.grid.TransformType.DCT2,
                                 fr.grid.TransformType.DCT2,
                                 fr.grid.TransformType.DST1)
                )

            b = fr.FieldVariable(
                mset,
                name="b", 
                long_name="buoyancy", 
                units="TODO", 
                is_spectral=is_spectral, 
                position=cell_center,
                transform_types=(fr.grid.TransformType.DST2,
                                 fr.grid.TransformType.DST2,
                                 fr.grid.TransformType.DST2)
                )

            field_list = [u, v, w, b]

            # add the fields from the custom field list
            for kw in mset.custom_fields:
                # Set default parameters if not provided
                if "position" not in kw:
                    # default position is cell center
                    kw["position"] = cell_center
                if "transform_types" not in kw:
                    # default transform type is DCT2
                    kw["transform_types"] = (fr.grid.TransformType.DCT2,
                                             fr.grid.TransformType.DCT2,
                                             fr.grid.TransformType.DCT2)
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
        # shortcuts
        pos = self.grid.cell_center
        interp = self.grid.interpolate
        dsqr = self.mset.dsqr
        u = self.u.arr
        v = self.v.arr
        w = self.w.arr

        ekin = 0.5*(u**2 + v**2 + dsqr*w**2)
        
        return fr.FieldVariable(
            self.mset, 
            is_spectral=self.is_spectral, 
            name="ekin", 
            long_name="Kinetic Energy",
            units="m^2/s^2",
            arr=ekin, 
            position=pos)

    @property
    def epot(self) -> fr.FieldVariable:
        r"""
        The potential energy

        .. math::
            E_{pot} = \frac{1}{2} \frac{b^2}{N^2}
        """
        # shortcuts
        pos = self.grid.cell_center
        N2 = self.mset.N2
        b = self.b.arr

        epot = 0.5*(b**2 / N2)

        return fr.FieldVariable(
            self.mset, 
            is_spectral=self.is_spectral,
            name="epot", 
            long_name="Potential Energy",
            units="m^2/s^2",
            arr=epot, 
            position=pos)
    
    @property
    def etot(self) -> fr.FieldVariable:
        """
        The total energy

        .. math::
            E_{tot} = E_{kin} + E_{pot}
        """
        etot = self.ekin + self.epot
        return fr.FieldVariable(
            self.mset, 
            is_spectral=self.is_spectral,
            name="etot", 
            long_name="Total Energy",
            units="m^2/s^2",
            arr=etot.arr,
            position=etot.position)

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
        """
        X-component of the relative vorticity

        .. math::
            \zeta_x = \delta^2 \partial_y w - \partial_z v
        """
        # shortcuts
        dsqr = self.mset.dsqr
        interp = self.grid.interpolate
        diff = self.grid.diff

        dwdy = diff(self.w.arr, axis=1, type="forward")
        dwdy_pos = self.w.position.shift(axis=1, direction="forward")

        dvdz = diff(self.v.arr, axis=2, type="forward")
        dvdz_pos = self.v.position.shift(axis=2, direction="forward")

        # if the positions do not match, interpolate the fields to cell centers
        if dwdy_pos != dvdz_pos:
            pos = self.grid.cell_center
            dwdy = interp(dwdy, dwdy_pos, pos)
            dvdz = interp(dvdz, dvdz_pos, pos)
        else:
            pos = dwdy_pos
        
        rel_vort_x = dsqr*dwdy - dvdz
            
        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            is_spectral=False, 
            name="vort_x", 
            long_name="x component of relative vorticity",
            units="1/s",
            arr=rel_vort_x,
            position=pos)

    @property
    def rel_vort_y(self) -> fr.FieldVariable:
        """
        Y-component of the relative vorticity

        .. math::
            \zeta_y = \partial_z u - \delta^2 \partial_x w
        """
        # shortcuts
        dsqr = self.mset.dsqr
        interp = self.grid.interpolate
        diff = self.grid.diff

        dudz = diff(self.u.arr, axis=2, type="forward")
        dudz_pos = self.u.position.shift(axis=2, direction="forward")

        dwdx = diff(self.w.arr, axis=0, type="forward")
        dwdx_pos = self.w.position.shift(axis=0, direction="forward")

        # if the positions do not match, interpolate the fields to cell centers
        if dudz_pos != dwdx_pos:
            pos = self.grid.cell_center
            dudz = interp(dudz, dudz_pos, pos)
            dwdx = interp(dwdx, dwdx_pos, pos)
        else:
            pos = dudz_pos
        
        rel_vort_y = dudz - dsqr*dwdx
            
        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            is_spectral=False, 
            name="vort_y", 
            long_name="y component of relative vorticity",
            units="1/s",
            arr=rel_vort_y,
            position=pos)

    @property
    def rel_vort_z(self) -> fr.FieldVariable:
        """
        Z-component of the relative vorticity (Horizontal Vorticity)

        .. math::
            \zeta_z = \partial_x v - \partial_y u
        """
        # shortcuts
        interp = self.grid.interpolate
        diff = self.grid.diff

        dvdx = diff(self.v.arr, axis=0, type="forward")
        dvdx_pos = self.v.position.shift(axis=0, direction="forward")

        dudy = diff(self.u.arr, axis=1, type="forward")
        dudy_pos = self.u.position.shift(axis=1, direction="forward")

        # if the positions do not match, interpolate the fields to cell centers
        if dvdx_pos != dudy_pos:
            pos = self.grid.cell_center
            dvdx = interp(dvdx, dvdx_pos, pos)
            dudy = interp(dudy, dudy_pos, pos)
        else:
            pos = dvdx_pos
        
        rel_vort_z = dvdx - dudy
            
        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            is_spectral=False, 
            name="vort_z", 
            long_name="horizontal vorticity",
            units="1/s",
            arr=rel_vort_z,
            position=pos)

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
        interp = self.grid.interpolate

        # calculate the horizontal vorticity
        ver_vort_x = self.rel_vort_x * Ro
        ver_vort_y = self.rel_vort_y * Ro
        ver_vort_z = self.rel_vort_z * Ro

        # calculate the buoyancy gradient
        buo_grad_x, buo_grad_y, buo_grad_z = self.grid.grad(self.b.arr * Ro)

        # interpolate the buoyancy gradient to the voriticities
        buo_grad_x_pos = self.b.position.shift(axis=0, direction="forward")
        buo_grad_y_pos = self.b.position.shift(axis=1, direction="forward")
        buo_grad_z_pos = self.b.position.shift(axis=2, direction="forward")
        buo_grad_x = interp(buo_grad_x, buo_grad_x_pos, ver_vort_x.position)
        buo_grad_y = interp(buo_grad_y, buo_grad_y_pos, ver_vort_y.position)
        buo_grad_z = interp(buo_grad_z, buo_grad_z_pos, ver_vort_z.position)

        x_part = ver_vort_x * buo_grad_x
        y_part = ver_vort_y * buo_grad_y
        z_part = (f0 + ver_vort_z) * (N2 + buo_grad_z)

        # interpolate the fields to the cell center
        x_part = interp(x_part, x_part.position, self.grid.cell_center)
        y_part = interp(y_part, y_part.position, self.grid.cell_center)
        z_part = interp(z_part, z_part.position, self.grid.cell_center)

        pot_vort = x_part + y_part + z_part

        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            is_spectral=False, 
            name="pot_vort",
            long_name="Potential Vorticity",
            units="TODO",
            arr=pot_vort,
            position=self.grid.cell_center)

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
        interp = self.grid.interpolate

        hor_vort = self.rel_vort_z
        hor_vort = interp(hor_vort, hor_vort.position, self.grid.cell_center)

        dbdz = self.grid.diff(self.b.arr, axis=2, type="forward")
        dbdz_pos = self.b.position.shift(axis=2, direction="forward")
        dbdz = interp(dbdz, dbdz_pos, self.grid.cell_center)

        pot_vort = Ro * (f0/N2 * dbdz + hor_vort)
        return fr.FieldVariable(
            self.mset,
            is_spectral=self.is_spectral,
            name="linear PV",
            long_name="Linear Potential Vorticity",
            units="TODO",
            arr=pot_vort,
            position=self.grid.cell_center)

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

        hor_vort = self.rel_vort_z
        local_Ro = Ro * hor_vort / f

        # calculate the local Rossby number
        return fr.FieldVariable(
            self.mset,
            is_spectral=self.is_spectral,
            name="loc Ro",
            long_name="Local Rossby Number",
            units="1",
            arr=local_Ro,
            position=local_Ro.position)

    # ----------------------------------------------------------------
    #  CFL numbers
    # ----------------------------------------------------------------

    @property
    def cfl_h(self) -> fr.FieldVariable:
        r"""
        The horizontal CFL number.

        .. math::
            CFL_h = \sqrt{u^2 + v^2} \frac{\Delta t}{\Delta}

        where :math:`\Delta t` is the time step and :math:`\Delta` is a 
        horizontal grid spacing:

        .. math::
            \Delta = \min(\Delta x, \Delta y)

        Returns:
            cfl_h (FieldVariable)  : Horizontal CFL number.
        """
        # shortcuts
        u = self.u; v = self.v
        dx, dy, dz = self.grid.dx
        dx = min(dx, dy)
        dt = self.mset.time_stepper.dt

        cfl_h = (u**2 + v**2)**0.5 * dt / dx

        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            arr=cfl_h.arr,
            is_spectral=self.is_spectral, 
            name="hor CFL",
            long_name="Horizontal CFL",
            position=self.grid.cell_center)

    @property
    def cfl_v(self) -> fr.FieldVariable:
        r"""
        Vertical CFL number.

        .. math::
            CFL_v = \sqrt{w^2} \frac{\Delta t}{\Delta z}

        where :math:`\Delta t` is the time step and :math:`\Delta z` is a 
        vertical grid spacing.

        Returns:
            cfl_v (FieldVariable)  : Vertical CFL number.
        """
        # shortcuts
        w = self.w
        dx, dy, dz = self.grid.dx
        dt = self.mset.time_stepper.dt

        cfl_v = fr.config.ncp.abs(w.arr) * dt / dz

        # Create the field variable
        return fr.FieldVariable(
            self.mset, 
            arr=cfl_v,
            is_spectral=self.is_spectral, 
            name="ver CFL",
            long_name="Vertical CFL",
            position=w.position)

fr.utils.jaxify_class(State)


class DiagnosticState(fr.StateBase):
    def __init__(self, mset: nh.ModelSettings, is_spectral=False, field_list=None) -> None:
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

fr.utils.jaxify_class(DiagnosticState)
