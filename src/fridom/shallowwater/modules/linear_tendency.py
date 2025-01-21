"""Linear tendency module for the shallow water model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr
import fridom.shallowwater as sw


@partial(fr.utils.jaxify, dynamic=("csqr", "f_coriolis"))
class LinearTendency(fr.modules.Module):

    r"""
    Computes the linear tendency of the shallow water model.

    The linear tendency is given by:

    .. math::
        \partial_t \boldsymbol{u} = f \underset{\neg}{\boldsymbol{v}} - \nabla p
        ~, \quad
        \partial_t p = -c^2 \nabla \cdot \boldsymbol{u}

    """

    name = "Linear Tendency"

    def _on_setup(self) -> None:
        self.f_coriolis = self.mset.f_coriolis
        self.csqr = self.mset.csqr

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self.linear_tendency(mz.z, mz.dz)
        return mz

    @fr.utils.jaxjit
    def linear_tendency(self, z: sw.State, dz: sw.State) -> sw.State:
        """Compute the linear tendency term."""
        interp = self.interp_module.interpolate
        diff = self.diff_module.diff
        div = self.diff_module.div

        # interpolate the coriolis parameter to the u position
        f = interp(self.f_coriolis, z.u.position)

        # calculate u-tendency
        dz.u +=   interp(z.v, z.u.position) * f - diff(z.p, axis=0)
        dz.v += - interp(z.u * f, z.v.position) - diff(z.p, axis=1)
        dz.p += - self.csqr * div((z.u, z.v))

        return dz
