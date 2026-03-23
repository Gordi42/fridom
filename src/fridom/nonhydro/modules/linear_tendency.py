"""Linear Tendency Module of the nonhydrostatic model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("f_coriolis", "N2", "dsqr"))
class LinearTendency(fr.modules.Module):

    """Computes the linear tendency of the nonhydrostatic model."""

    name = "Linear Tendency"

    def _on_setup(self) -> None:
        self.f_coriolis = self.mset.f_coriolis
        self.N2 = self.mset.N2_field
        self.dsqr = self.mset.dsqr

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        interp = self.interp_module.interpolate
        z = mz.z

        # interpolate the coriolis parameter to the u position
        f = interp(self.f_coriolis, z.u.position)

        # calculate u-tendency
        mz.dz.u +=   interp(z.v, z.u.position) * f
        mz.dz.v += - interp(z.u * f, z.v.position)
        mz.dz.w +=   interp(z.b, z.w.position) / self.dsqr
        mz.dz.b += - interp(z.w, z.b.position) * self.N2

        return mz
