import fridom.framework as fr


@fr.utils.jaxify
class CenteredAdvection(fr.modules.Module):
    """
    Centered advection scheme.

    Description
    -----------
    Let :math:`\\mathbf{v}` be the velocity field and :math:`q` be the quantity
    to be advected. For a divergence-free velocity field, the advection term
    can be written as:

    .. math::
        \\mathbf{v} \\cdot \\nabla q = \\nabla \\cdot (\\mathbf{v} q)

    Lets consider the :math:`x`-component of the flux 
    :math:`\\mathbf{F}=\\mathbf{v} q`. The flux divergence 
    :math:`\\partial_x F_x` is calculated using forward or backward differences.
    For that the flux is interpolated to the cell faces of the quantity :math:`q`.

    Parameters
    ----------
    `diff` : `DiffBase | None`, (default=None)
        Differentiation module to use.
        If None, the differentiation module of the grid is used.
    `interpolation` : `InterpolationBase | None`, (default=None)
        The interpolation module to use.
        If None, the interpolation module of the grid is used.
    """
    def __init__(self):
        super().__init__(name="Centered Advection")
        self.required_halo = 2
        return

    @fr.utils.jaxjit
    def flux_divergence(self, 
                        velocity: 'tuple[fr.FieldVariable]',
                        quantity: 'fr.FieldVariable') -> 'fr.FieldVariable':
        # shorthand notation
        inter = self.interp_module.interpolate
        diff = self.diff_module.diff
        Ro = self.mset.Ro
        q_pos = quantity.position

        res = fr.FieldVariable(**quantity.get_kw())

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = q_pos.shift(axis)
            flux = inter(v, flux_pos) * inter(quantity, flux_pos)
            res += diff(flux, axis, order=1)
        return Ro * res

    @fr.utils.jaxjit
    def advect_state(self, 
                     z: nh.State, 
                     dz: nh.State, 
                     velocity: tuple[fr.FieldVariable]) -> nh.State:
        # calculate the advection term
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            dz.fields[name] -= self.flux_divergence(velocity, quantity)
        return dz


    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> None:
        """
        Compute the advection term of the state vector z.
        """
        # calculate the full velocity field
        # in future, this enables the use of a background velocity field
        zf = mz.z
        velocity = (zf.u, zf.v, zf.w)

        mz.dz = self.advect_state(zf, mz.dz, velocity)

        return mz
