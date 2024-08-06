import fridom.framework as fr
import fridom.nonhydro as nh
from functools import partial


@partial(fr.utils.jaxify, dynamic=("f_coriolis", ))
class LinearTendency(fr.modules.Module):
    """
    This class computes the linear tendency of the model.
    """
    def __init__(self, interpolation: fr.grid.InterpolationModule | None = None):
        super().__init__(name="Linear Tendency")
        self.interpolation = interpolation
        self.f_coriolis = None

    @fr.modules.setup_module
    def setup(self):
        if self.interpolation is None:
            self.interpolation = self.mset.grid._interp_mod
        else:
            self.interpolation.setup(mset=self.mset)
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
        interp = self.interpolation.interpolate

        # interpolate the coriolis parameter to the u position
        f = interp(self.mset.f_coriolis, z.u.position)

        # calculate u-tendency
        dz.u +=   interp(z.v, z.u.position) * f
        dz.v += - interp(z.u * f, z.v.position)
        dz.w +=   interp(z.b, z.w.position) / self.mset.dsqr
        dz.b += - interp(z.w, z.b.position) * self.mset.N2

        return dz

    @property
    def required_halo(self) -> int:
        if self.interpolation is None:
            # while the interpolation module is not set up, we cannot determine
            # the required halo
            return 0
        else:
            return self.interpolation.required_halo
    
    @required_halo.setter
    def required_halo(self, value: int):
        return