import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class PressureGradientTendency(fr.modules.Module):
    """
    This class computes the pressure gradient tendency of the model.
    """
    name = "Pressure Gradient"

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.pressure_gradient_tendency(mz.z_diag.p, mz.dz)
        return mz

    @fr.utils.jaxjit
    def pressure_gradient_tendency(
            self, p: fr.FieldVariable, dz: nh.State) -> nh.State:
        """Compute the pressure gradient tendency of the model."""
        # compute gradient of pressure
        p_grad = self.diff_module.grad(p)

        # remove the gradient from the velocity tendency
        dz.u -= p_grad[0]
        dz.v -= p_grad[1]
        dz.w -= p_grad[2] / self.mset.dsqr

        return dz
