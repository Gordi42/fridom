import fridom.framework as fr


@fr.utils.jaxify
class CenteredAdvection(fr.modules.advection.AdvectionBase):
    name = "Centered Advection"

    @fr.utils.jaxjit
    def advection(self, 
                  velocity: 'tuple[fr.FieldVariable]',
                  quantity: 'fr.FieldVariable') -> 'fr.FieldVariable':
        # shorthand notation
        inter = self.interp_module.interpolate
        diff = self.diff_module.diff
        q_pos = quantity.position

        res = fr.FieldVariable(**quantity.get_kw())

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = q_pos.shift(axis)
            flux = inter(v, flux_pos) * inter(quantity, flux_pos)
            res -= diff(flux, axis, order=1)
        return res
