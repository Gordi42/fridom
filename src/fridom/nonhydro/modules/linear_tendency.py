import fridom.framework as fr
import fridom.nonhydro as nh
from functools import partial


@partial(fr.utils.jaxify, dynamic=("f_coriolis", ))
class LinearTendency(fr.modules.Module):
    """
    This class computes the linear tendency of the model.
    """
    name = "Linear Tendency"
    def __init__(self):
        super().__init__()
        self.f_coriolis = None

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        self.f_coriolis = self.mset.f_coriolis
        return

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.linear_tendency(mz.z, mz.dz)
        return mz

    @fr.utils.jaxjit
    def linear_tendency(self, z: nh.State, dz: nh.State) -> nh.State:
        """
        Compute the linear tendency of the model.
        """
        interp = self.interp_module.interpolate

        # interpolate the coriolis parameter to the u position
        f = interp(self.f_coriolis, z.u.position)

        # calculate u-tendency
        dz.u +=   interp(z.v, z.u.position) * f
        dz.v += - interp(z.u * f, z.v.position)
        dz.w +=   interp(z.b, z.w.position) / self.mset.dsqr
        dz.b += - interp(z.w, z.b.position) * self.mset.N2

        return dz
