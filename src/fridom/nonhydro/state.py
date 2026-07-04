"""The state vector class for the nonhydrostatic model."""
from __future__ import annotations

from collections import OrderedDict

import fridom.framework as fr
import fridom.nonhydro as nh

NEUMANN = fr.grid.BCType.NEUMANN
DIRICHLET = fr.grid.BCType.DIRICHLET

@fr.utils.jaxify
class State(fr.VectorField):

    """
    The state vector class for the nonhydrostatic model.

    Description
    -----------
    The default scalar fields of the state vector are:

    - u: Velocity in the x-direction.
    - v: Velocity in the y-direction.
    - w: Velocity in the z-direction.
    - b: Buoyancy.

    A variety of diagnostic fields can be calculated from the state vector,
    such as the kinetic energy, potential energy, total energy, relative
    vorticity, potential vorticity, and the local Rossby number.

    """

    def __init__(self, mset: nh.ModelSettings, **kwargs: any) -> None:
        super().__init__(mset, **kwargs)
        # we set the class to State, so that child classes will always
        # be of type State
        self.__class__ = State

    @staticmethod
    def _create_default_fields(mset: nh.ModelSettings,
                               vector_dim: int | None,  # noqa: ARG004
                               **kwargs: any,
                               ) -> OrderedDict[str, fr.ScalarField]:
        cell_center = mset.grid.cell_center

        u = fr.ScalarField(
            mset,
            name="u",
            long_name="u - velocity",
            units="m/s",
            position=cell_center.shift(axis=0),
            bc_types=(DIRICHLET, NEUMANN, NEUMANN),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        v = fr.ScalarField(
            mset,
            name="v",
            long_name="v - velocity",
            units="m/s",
            position=cell_center.shift(axis=1),
            bc_types=(NEUMANN, DIRICHLET, NEUMANN),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        w = fr.ScalarField(
            mset,
            name="w",
            long_name="w - velocity",
            units="m/s",
            position=cell_center.shift(axis=2),
            bc_types=(NEUMANN, NEUMANN, DIRICHLET),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        b = fr.ScalarField(
            mset,
            name="b",
            long_name="Buoyancy",
            units="m/s²",
            position=cell_center,
            bc_types=(NEUMANN, NEUMANN, DIRICHLET),
            flags={"ENABLE_MIXING": True},
            **kwargs)

        fields = OrderedDict([("u", u), ("v", v), ("w", w), ("b", b)])
        return State._add_custom_fields(
            mset, fields, mset.custom_state_fields, **kwargs)

    # ----------------------------------------------------------------
    #  State Variables
    # ----------------------------------------------------------------

    @property
    def u(self) -> fr.ScalarField:
        """Velocity in the x-direction."""
        return self.fields["u"]

    @u.setter
    def u(self, value: fr.ScalarField) -> None:
        self.fields["u"] = value

    @property
    def v(self) -> fr.ScalarField:
        """Velocity in the y-direction."""
        return self.fields["v"]

    @v.setter
    def v(self, value: fr.ScalarField) -> None:
        """Velocity in the y-direction."""
        self.fields["v"] = value

    @property
    def w(self) -> fr.ScalarField:
        """Velocity in the z-direction."""
        return self.fields["w"]

    @w.setter
    def w(self, value: fr.ScalarField) -> None:
        self.fields["w"] = value

    @property
    def b(self) -> fr.ScalarField:
        """Buoyancy."""
        return self.fields["b"]

    @b.setter
    def b(self, value: fr.ScalarField) -> None:
        self.fields["b"] = value

    @property
    def velocity(self) -> fr.VectorField:
        """The velocity vector field."""
        return self[:3]

    @property
    def tracers(self) -> fr.VectorField:
        """The tracer fields."""
        return self[3:]

    # ----------------------------------------------------------------
    #  Energy Variables
    # ----------------------------------------------------------------

    @property
    def ekin(self) -> fr.ScalarField:
        r"""
        The kinetic energy.

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
    def epot(self) -> fr.ScalarField:
        r"""
        The potential energy.

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
        if self.mset.stratification_n2 != 0:
            epot = 0.5*(self.b**2 / self.mset.stratification_n2_field)
        else:
            epot = self.b * self.grid.x_mesh[2]

        # Set the attributes
        epot.name = "epot"
        epot.long_name = "Potential Energy"
        epot.units = "m²/s²"
        epot.position = self.grid.cell_center
        return epot

    @property
    def etot(self) -> fr.ScalarField:
        r"""
        The total energy.

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
    def rel_vort(self) -> fr.VectorField:
        r"""
        The relative vorticity.

        .. math::
            \boldsymbol{\zeta} = \nabla \times \boldsymbol{u}
        """
        return fr.VectorField(self.mset, field_list=[self.rel_vort_x,
                                                     self.rel_vort_y,
                                                     self.rel_vort_z])

    @property
    def rel_vort_x(self) -> fr.ScalarField:
        r"""
        X-component of the relative vorticity.

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
    def rel_vort_y(self) -> fr.ScalarField:
        r"""
        Y-component of the relative vorticity.

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
    def rel_vort_z(self) -> fr.ScalarField:
        r"""
        Z-component of the relative vorticity (horizontal vorticity).

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
    def pot_vort(self) -> fr.ScalarField:
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
            msg = "Potential vorticity is not implemented for spectral fields."
            raise NotImplementedError(msg)

        # shortcuts
        f0 = self.mset.f0
        brunt_vaisala_n2 = self.mset.stratification_n2_field
        rossby_number = self.mset.rossby_number

        # calculate the horizontal vorticity
        ver_vort_x = self.rel_vort_x * rossby_number
        ver_vort_y = self.rel_vort_y * rossby_number
        ver_vort_z = self.rel_vort_z * rossby_number

        # calculate the buoyancy gradient
        buo_grad_x, buo_grad_y, buo_grad_z = (self.b * rossby_number).grad()

        # interpolate the buoyancy gradient to the voriticities
        buo_grad_x = buo_grad_x.interpolate(ver_vort_x.position)
        buo_grad_y = buo_grad_y.interpolate(ver_vort_y.position)
        buo_grad_z = buo_grad_z.interpolate(ver_vort_z.position)

        # Calculate each component of the potential vorticity
        x_part = ver_vort_x * buo_grad_x
        y_part = ver_vort_y * buo_grad_y
        z_part = (ver_vort_z + f0) * (brunt_vaisala_n2 + buo_grad_z)

        pot_vort = x_part + y_part + z_part

        # Set the attributes
        pot_vort.name = "pot_vort"
        pot_vort.long_name = "Potential Vorticity"
        pot_vort.units = "n/a"
        pot_vort.position = self.grid.cell_center

        return pot_vort

    @property
    def linear_pot_vort(self) -> fr.ScalarField:
        r"""
        Linearized potential vorticity.

        .. math::
            Q = Ro \left( \frac{f}{N^2} \partial_z b + \zeta_z \right)

        where :math:`Ro` is the Rossby number, :math:`f` is the Coriolis
        parameter, :math:`N^2` is the buoyancy frequency, :math:`b` is the
        buoyancy field, and :math:`\zeta_z` is the vertical component of the
        relative vorticity.
        """
        # shortcuts
        f0 = self.mset.f0
        brunt_vaisala_n2 = self.mset.stratification_n2_field
        rossby_number = self.mset.rossby_number

        hor_vort = self.rel_vort_z.interpolate(self.grid.cell_center)
        dbdz = self.b.diff(axis=2).interpolate(self.grid.cell_center)
        pot_vort = rossby_number * (f0/brunt_vaisala_n2 * dbdz + hor_vort)

        # Set the attributes
        pot_vort.name = "linear pot vort"
        pot_vort.long_name = "Linear Potential Vorticity"
        pot_vort.units = "n/a"
        pot_vort.position = self.grid.cell_center

        return pot_vort

    @property
    def local_rossby_number(self) -> fr.ScalarField:
        r"""
        Local Rossby number.

        .. math::
            Ro_\text{local} = Ro \, \frac{\zeta_z}{f_0}

        where :math:`Ro` is the Rossby number, :math:`\zeta_z` is the vertical
        component of the relative vorticity, and :math:`f_0` is the Coriolis
        parameter.
        """
        # shortcuts
        f_coriolis = self.mset.f_coriolis
        rossby_number = self.mset.rossby_number

        local_rossby_number = rossby_number * self.rel_vort_z / f_coriolis

        # Set the attributes
        local_rossby_number.name = "loc Ro"
        local_rossby_number.long_name = "Local Rossby Number"
        local_rossby_number.units = "1"

        return local_rossby_number

    # ----------------------------------------------------------------
    #  CFL numbers
    # ----------------------------------------------------------------

    @property
    def cfl(self) -> fr.ScalarField:
        r"""
        The CFL number.

        .. math::
            CFL = \max\left\{ \frac{u}{\Delta x}, \frac{v}{\Delta y},
                              \frac{w}{\Delta z} \right\} \Delta t

        where :math:`\Delta t` is the time step and :math:`\Delta x` is the
        grid spacing.

        Returns
        -------
            cfl (ScalarField)  : Horizontal CFL number.

        """
        dx, dy, dz = self.grid.dx
        dt = self.mset.time_stepper.dt
        cfl_u = self.u.abs() * dt / dx
        cfl_v = self.v.abs() * dt / dy
        cfl_w = self.w.abs() * dt / dz

        cfl = fr.config.ncp.maximum(cfl_u.arr, cfl_v.arr)
        cfl = fr.config.ncp.maximum(cfl, cfl_w.arr)

        # Create the scalar field
        return fr.ScalarField(
            self.mset,
            arr=cfl,
            is_spectral=self.is_spectral,
            name="cfl",
            long_name="CFL Number",
            position=self.grid.cell_center)


@fr.utils.jaxify
class DiagnosticState(fr.VectorField):

    """
    The diagnostic state vector class for the nonhydrostatic model.

    Description
    -----------
    The default scalar fields of the diagnostic state vector are:

    - p: Pressure.
    - div: Divergence.

    """

    @staticmethod
    def _create_default_fields(mset: fr.ModelSettingsBase,
                               vector_dim: int | None,  # noqa: ARG004
                               **kwargs: any,
                               ) -> OrderedDict[str, fr.ScalarField]:
        p = fr.ScalarField(
            mset,
            name="p",
            long_name="Pressure",
            units="m²/s",
            position=mset.grid.cell_center,
            **kwargs)

        div = fr.ScalarField(
            mset,
            name="div",
            long_name="Divergence",
            units="1/s",
            position=mset.grid.cell_center,
            **kwargs)

        fields = OrderedDict([("p", p), ("div", div)])
        return DiagnosticState._add_custom_fields(
            mset, fields, mset.custom_diagnostic_fields, **kwargs)

    @property
    def p(self) -> fr.ScalarField:
        """The pressure field."""
        return self.fields["p"]

    @p.setter
    def p(self, value: fr.ScalarField) -> None:
        self.fields["p"] = value

    @property
    def div(self) -> fr.ScalarField:
        """The divergence field."""
        return self.fields["div"]

    @div.setter
    def div(self, value: fr.ScalarField) -> None:
        self.fields["div"] = value
