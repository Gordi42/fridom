import fridom.framework as fr
from functools import partial


class CenteredAdvection(fr.modules.Module):
    def __init__(self, 
                 diff_mod: 'fr.grid.DiffModule | None' = None,
                 interp_mod: 'fr.grid.InterpolationModule | None' = None):
        super().__init__(name="Centered Advection")
        self.required_halo = 1
        self.diff_module = diff_mod
        self.interp_module = interp_mod
        return

    @fr.modules.setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff_module is None:
            self.diff_module = self.mset.grid.diff_mod
        else:
            self.diff_module.setup(mset=self.mset)

        # setup the interpolation modules
        if self.interp_module is None:
            self.interp_module = self.mset.grid.interp_mod
        else:
            self.interp_module.setup(mset=self.mset)
        return

    @fr.utils.jaxjit
    def flux_divergence(self, 
                        velocity: 'tuple[fr.FieldVariable]',
                        quantity: 'fr.FieldVariable') -> 'fr.FieldVariable':
        # shorthand notation
        inter = self.interpolation.interpolate
        scaling = self.mset.nonlinear_scaling
        q_pos = quantity.position

        res = fr.FieldVariable(**quantity.get_kw())

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = q_pos.shift(axis)
            flux = inter(v, flux_pos) * inter(quantity, flux_pos)
            res += flux.diff(axis, order=1)
        return scaling * res
    
    @fr.utils.jaxjit
    def advect_state(self, 
                     z: fr.StateBase, 
                     dz: fr.StateBase, 
                     velocity: tuple[fr.FieldVariable]) -> fr.StateBase:
        """
        Advects all fields in the state `z` that do not have the flag "NO_ADV".
        """
        # calculate the advection term
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            dz.fields[name] -= self.flux_divergence(velocity, quantity)
        return dz

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> None:
        # calculate the full velocity field
        # in future, this enables the use of a background velocity field
        zf = mz.z
        velocity = (zf.u, zf.v, zf.w)

        mz.dz = self.advect_state(zf, mz.dz, velocity)

        return mz


    # ================================================================
    #  Properties
    # ================================================================
    
    @property
    def diff_module(self) -> fr.grid.DiffModule:
        """The differentiation module."""
        return self._diff_module
    
    @diff_module.setter
    def diff_module(self, value):
        self._diff_module = value
        return

    @property
    def interp_module(self) -> fr.grid.InterpolationModule:
        """The interpolation module to interpolate the diffusion coefficients."""
        return self._interp_module

    @interp_module.setter
    def interp_module(self, value):
        self._interp_module = value
        return

    @property
    def background_state(self) -> fr.StateBase:
        """The background state."""
        return self._background_state