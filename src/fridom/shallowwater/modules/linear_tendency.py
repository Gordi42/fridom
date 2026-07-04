"""Linear tendency module for the shallow water model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("csqr", "f_coriolis"))
class LinearTendency(fr.modules.Module):

    r"""
    Computes the linear tendency of the shallow water model.

    The linear tendency is given by:

    .. math::
        \partial_t \boldsymbol{u} =
            f \underset{\neg}{\boldsymbol{v}} - \nabla p
        ~, \quad
        \partial_t p = -c^2 \nabla \cdot \boldsymbol{u}

    """

    name = "Linear Tendency"

    def __init__(self) -> None:
        super().__init__()
        self.f_coriolis = None
        self.csqr = None

    def _on_setup(self) -> None:
        self.f_coriolis = self.mset.f_coriolis
        self.csqr = self.mset.csqr_field

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        interp = self.interp_module.interpolate
        diff = self.diff_module.diff
        div = self.diff_module.div
        z = mz.z

        # positions
        upos = z.u.position
        vpos = z.v.position

        c = self.csqr
        # interpolate the coriolis parameter to the u position
        f = interp(self.f_coriolis, z.u.position)

        # calculate u-tendency
        mz.dz.u +=   interp(z.v, upos) * f - diff(z.p, axis=0)
        mz.dz.v += - interp(z.u * f, vpos) - diff(z.p, axis=1)
        mz.dz.p += - div((interp(c, upos) * z.u, interp(c, vpos) * z.v))

        return mz
