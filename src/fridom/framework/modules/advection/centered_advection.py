import fridom.framework as fr


@fr.utils.jaxify
class CenteredAdvection(fr.modules.advection.AdvectionBase):
    r"""
    Centered advection scheme.

    Description
    -----------
    For the centered advection scheme, we assume that the velocity field is
    divergence-free. The advection term can then be written as:

    .. math::
        \mathcal{A}(\boldsymbol{v}, q) = -\boldsymbol{v} \cdot \nabla q = 
            - \nabla \cdot (\boldsymbol{v} q)

    where :math:`q` is the quantity to be advected and :math:`\boldsymbol{v}` 
    is the velocity field. The flux divergence :math:`\nabla \cdot (\boldsymbol{v} q)`
    is calculated using forward or backward differences. For that the flux is
    interpolated to the cell faces of the quantity :math:`q`:

    ::

                    Position of the quantity q
                                ↓
        |       x       |       x       |       x       |
                        ↑
            Position of the flux Fx

    """
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
