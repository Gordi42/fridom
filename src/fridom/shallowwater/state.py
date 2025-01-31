"""The state vector class for the shallow water model."""
from __future__ import annotations

from collections import OrderedDict

import fridom.framework as fr
import fridom.shallowwater as sw

NEUMANN = fr.grid.BCType.NEUMANN
DIRICHLET = fr.grid.BCType.DIRICHLET

@fr.utils.jaxify
class State(fr.VectorField):

    r"""
    State vector of the 2D shallow water model.

    Description
    -----------
    The default scalar fields of the state vector are:

    - u: Velocity in the x-direction.
    - v: Velocity in the y-direction.
    - p: Pressure field, with :math:`p = g \eta`, where :math:`\eta` is the
    free surface elevation.

    """

    def __init__(self, mset: sw.ModelSettings, **kwargs: any) -> None:
        super().__init__(mset, **kwargs)
        # we set the class to State, so that child classes will always be of type State
        self.__class__ = State

    @staticmethod
    def _create_default_fields(mset: sw.ModelSettings,
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
            bc_types=(DIRICHLET, NEUMANN),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        v = fr.ScalarField(
            mset,
            name="v",
            long_name="v - velocity",
            units="m/s",
            position=cell_center.shift(axis=1),
            bc_types=(NEUMANN, DIRICHLET),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        p = fr.ScalarField(
            mset,
            name="p",
            long_name="pressure",
            units="m²/s²",
            position=cell_center,
            bc_types=(DIRICHLET, DIRICHLET),
            **kwargs)

        # TODO(Silvano): add the custom fields from model settings

        fields = OrderedDict([("u", u), ("v", v), ("p", p)])
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
        self.fields["v"] = value

    @property
    def p(self) -> fr.ScalarField:
        r"""
        Pressure :math:`p`.

        Description
        -----------
        The pressure field is defined as

        .. math::
            p = g \eta

        where :math:`\eta` is the free surface elevation and :math:`g` is the
        gravity acceleration.
        """
        return self.fields["p"]

    @p.setter
    def p(self, value: fr.ScalarField) -> None:
        self.fields["p"] = value

    @property
    def velocity(self) -> fr.VectorField:
        """Velocity vector."""
        return fr.VectorField(self.mset, field_list=[self.u, self.v])

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
        Vertically integrated kinetic energy.

        .. math::
            E_{\text{kin}} = \frac{Ro^2}{2} h_{\text{full}} (u^2 + v^2)

        with

        .. math::
            h_{\text{full}} = c^2 + Ro p

        Note:
            The energy is scaled with the gravity acceleration g.

        """
        sw.exceptions.FieldSpaceError.check_if_physical(self)

        csqr = self.mset.csqr_field
        rossby_number = self.mset.Ro
        h_full = csqr + rossby_number * self.p
        ekin = 0.5 * rossby_number**2 * h_full * (self.u**2 + self.v**2)

        # Set the attributes
        ekin.name = "ekin"
        ekin.long_name = "Kinetic Energy"
        ekin.units = "?"
        ekin.position = self.grid.cell_center
        return ekin

    @property
    def epot(self) -> fr.ScalarField:
        r"""
        Vertically integrated kinetic energy.

        .. math::
            E_{\text{pot}} = \frac{1}{2} h_{\text{full}}^2

        with

        .. math::
            h_{\text{full}} = c^2 + Ro p

        Note:
            The energy is scaled with the gravity acceleration g.

        """
        sw.exceptions.FieldSpaceError.check_if_physical(self)

        csqr = self.mset.csqr_field
        rossby_number = self.mset.Ro
        h_full = csqr + rossby_number * self.p
        epot = 0.5 * h_full ** 2

        # Set the attributes
        epot.name = "epot"
        epot.long_name = "Potential Energy"
        epot.units = "?"
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
        etot.units = "?"
        etot.position = self.grid.cell_center
        return etot

    @property
    def spectral_ekin(self) -> fr.ScalarField:
        r"""
        Spectral kinetic energy density.

        .. math::
            S_{\text{kin}} = \frac{1}{2} (|\hat{u}|^2 + |\hat{v}|^2)

        """
        sw.exceptions.FieldSpaceError.check_if_spectral(self)

        ekin = 0.5 * (self.u * self.u.conj() + self.v * self.v.conj())

        # Set the attributes
        ekin.name = "spectral_ekin"
        ekin.long_name = "Spectral Kinetic Energy Density"
        ekin.units = "?"
        ekin.position = self.grid.cell_center
        return ekin

    # ----------------------------------------------------------------
    #  Vorticity
    # ----------------------------------------------------------------

    @property
    def rel_vort(self) -> fr.ScalarField:
        r"""
        Relative vorticity.

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
    def pot_vort(self) -> fr.ScalarField:
        r"""
        Scaled potential vorticity field.

        .. math::
            Q = \frac{\zeta + f \right}{c^2 + Ro p}

        where :math:`f` is the Coriolis parameter, and :math:`\zeta` is the
        relative vorticity.
        """
        if self.is_spectral:
            msg = "Potential vorticity is not implemented for spectral fields."
            raise NotImplementedError(msg)

        # shortcuts
        f = self.mset.f_coriolis
        csqr = self.mset.csqr_field
        rossby_number = self.mset.Ro

        pot_vort = (self.rel_vort + f) / (csqr + rossby_number * self.p)

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
        rossby_number = self.mset.Ro

        local_rossby_number = rossby_number * self.rel_vort / f_coriolis

        # Set the attributes
        local_rossby_number.name = "loc Ro"
        local_rossby_number.long_name = "Local Rossby Number"
        local_rossby_number.units = "1"

        return local_rossby_number

    @property
    def cfl(self) -> fr.ScalarField:
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

        # Create the scalar field
        return fr.ScalarField(
            self.mset,
            arr=cfl,
            is_spectral=self.is_spectral,
            name="cfl",
            long_name="CFL Number",
            position=self.grid.cell_center)
