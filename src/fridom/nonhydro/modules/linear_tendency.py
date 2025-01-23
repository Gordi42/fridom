"""Linear Tendency Module of the nonhydrostatic model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("f_coriolis",))
class LinearTendency(fr.modules.Module):

    """Computes the linear tendency of the nonhydrostatic model."""

    name = "Linear Tendency"

    def _on_setup(self) -> None:
        self.f_coriolis = self.mset.f_coriolis

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self.linear_tendency(mz.z, mz.dz)
        return mz

    @fr.utils.jaxjit
    def linear_tendency(self, z: nh.State, dz: nh.State) -> nh.State:
        """Compute the linear tendency of the model."""
        interp = self.interp_module.interpolate

        # interpolate the coriolis parameter to the u position
        f = interp(self.f_coriolis, z.u.position)

        # calculate u-tendency
        dz.u +=   interp(z.v, z.u.position) * f
        dz.v += - interp(z.u * f, z.v.position)
        dz.w +=   interp(z.b, z.w.position) / self.mset.dsqr
        dz.b += - interp(z.w, z.b.position) * self.mset.N2

        return dz
