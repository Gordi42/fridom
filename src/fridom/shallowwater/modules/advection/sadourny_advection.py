"""Sadourny advection scheme for the shallow water equations."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr
import fridom.shallowwater as sw


@partial(fr.utils.jaxify, dynamic=("csqr", ))
class SadournyAdvection(fr.modules.advection.AdvectionBase):

    r"""
    Advection scheme based on Sadourny [1975] that conserves the total energy.

    The nonlinear advection terms for the shallow water equations are given by:

    .. math::
        \partial_t \boldsymbol{u}
                     = - (\boldsymbol{u} + \boldsymbol{u}_b)
                         \cdot \nabla \boldsymbol{u}
                     = - \underset{\neg}{\boldsymbol{u}} \zeta
                       - \frac{1}{2} \nabla \boldsymbol{u}^2
                       - \nabla \left(
                           \boldsymbol{u_b} \cdot \boldsymbol{u} \right)

        \partial_t p = - \nabla \left\[
            (\boldsymbol{u} + \boldsymbol{u}_b) p \right\]

        \partial_t C = - (\boldsymbol{u} + \boldsymbol{u}_b) \cdot \nabla C

    where :math:`\boldsymbol{u_b}` is a divergence free background flow
    that can be set with the `background` attribute of this module,
    :math:`\zeta` is the relative vorticity, and :math:`C` is a passive
    tracer.
    We express the rotational part of the momentum advection with the potential
    vorticity :math:`q`:

    .. math::
        \underset{\neg}{\boldsymbol{u}} \zeta
            = \underset{\neg}{\boldsymbol{f_u}} q

    with

    .. math::
        \boldsymbol{f_u} = (c^2 + Ro~ p) \boldsymbol{u}
        ~, \quad
        q = \frac{\zeta}{c^2 + Ro~ p}
    """

    name = "Sadourny Advection"

    def __init__(self) -> None:
        super().__init__()
        self.csqr = None

    def _on_setup(self) -> None:
        self._required_halo = 2
        self.csqr = self.mset.csqr

    def advect_state(self, z: sw.State, dz: sw.State) -> sw.State:  # noqa: D102
        if self.background is None and self.disable_nonlinear:
            return dz
        if self.disable_nonlinear:
            zf = self.background
        else:
            # Compute the full state vector (including the background)
            zf = z if self.background is None else z + self.background

        scale = self.scaling
        interp = self.interp_module.interpolate
        diff_mod = self.diff_module

        # ================================================================
        #  Define some grid positions
        # ================================================================

        center = z.p.position
        east = z.p.position.shift(0)
        north = z.p.position.shift(1)
        northeast = east.shift(1)

        # ----------------------------------------------------------------
        #  Compute the nonlinear term of the pressure tendency - ∇(vp)
        # ----------------------------------------------------------------
        fx = zf.u * interp(z.p, east)
        fy = zf.v * interp(z.p, north)
        dz.p -= scale * diff_mod.div((fx, fy))

        # ----------------------------------------------------------------
        #  Advection of passive tracer - v∇C = - ∇(vC) + C∇v
        # ----------------------------------------------------------------
        div = None
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            if name in ["u", "v", "p"]:
                continue
            # compute the divergence of the velocity field (only once)
            if div is None:
                div = diff_mod.div((zf.u, zf.v))

            fx = zf.u * interp(quantity, east)
            fy = zf.v * interp(quantity, north)

            df = - diff_mod.div((fx, fy)) + quantity * div
            dz.fields[name] -= scale * df

        # ----------------------------------------------------------------
        #  Advection of momentum
        # ----------------------------------------------------------------

        # start with momentum advection by the background flow
        if self.background is not None:
            u_b = self.background.u
            v_b = self.background.v
            # u-component
            fx = interp(u_b, center) * interp(z.u, center)
            fy = interp(v_b, northeast) * interp(z.u, northeast)
            dz.u -= scale * diff_mod.div((fx, fy))
            # v-component
            fx = interp(u_b, northeast) * interp(z.v, northeast)
            fy = interp(v_b, center) * interp(z.v, center)
            dz.v -= scale * diff_mod.div((fx, fy))

        # now do the nonlinear advection
        if self.disable_nonlinear:
            return dz

        # compute the potential vorticity
        zeta = z.rel_vort
        # check if we should use scale or Ro here
        h_full = self.csqr + scale * z.p
        q = zeta / interp(h_full, northeast)

        # interp set h_full to zero on boundaries, as a result values of q on
        # boundaries are nan. We set them to zero here.
        q.arr = fr.config.ncp.nan_to_num(q.arr, 0.0)

        # compute the fluxes fu and fv at the northeast position
        # (to match the vorticity)
        fu = interp(z.u * interp(h_full, east), northeast)
        fv = interp(z.v * interp(h_full, north), northeast)

        # compute the kinetic energy
        ekin = 0.5 * (interp(z.u**2, center) + interp(z.v**2, center))

        # compute the advection terms
        dz.u += scale * ( interp(fv * q, east ) - diff_mod.diff(ekin, axis=0))
        dz.v += scale * (-interp(fu * q, north) - diff_mod.diff(ekin, axis=1))
        return dz
