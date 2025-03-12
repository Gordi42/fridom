"""A module that computes the pressure gradient tendency of the model."""
from functools import partial

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("dsqr", ))
class PressureGradientTendency(fr.modules.Module):

    """Pressure gradient tendency module."""

    name = "Pressure Gradient"

    def _on_setup(self) -> None:
        self.dsqr = self.mset.dsqr

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self.pressure_gradient_tendency(mz.z_diag.p, mz.dz)
        return mz

    @fr.utils.jaxjit
    def pressure_gradient_tendency(
            self, p: fr.ScalarField, dz: nh.State) -> nh.State:
        """Compute the pressure gradient tendency of the model."""
        # compute gradient of pressure
        p_grad = self.diff_module.grad(p)

        # remove the gradient from the velocity tendency
        dz.u -= p_grad[0]
        dz.v -= p_grad[1]
        dz.w -= p_grad[2] / self.dsqr

        return dz
