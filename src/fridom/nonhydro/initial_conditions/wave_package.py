import fridom.nonhydro as nh

class WavePackage(nh.State):
    r"""
    Wave package initial condition.
    
    Description
    -----------
    Creates a polarized single wave (:py:class:`nh.initial_conditions.SingleWave`) 
    and applies a mask to it. The mask is a Gaussian function centered at 
    `mask_pos` with a width of `mask_width`:

    .. math::
        M(\boldsymbol x) = \prod_{i=1}^{3} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)
    
    where :math:`p_i` is the position and :math:`w_i` is the width of the mask
    in the :math:`i`-th direction. The final wave package is given by:

    .. math::
        z = \mathbf{P}_s \cdot \left( S(\boldsymbol{x}) M(\boldsymbol{x}) \right)
    
    where :math:`S(\boldsymbol{x})` is the single wave and :math:`\\mathbf{P}_s` 
    is the projection operator onto the mode `s`.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `mask_pos` : `tuple[float | None]`
        The position of the mask in the x, y, and z directions.
        If `None`, the mask is not applied in that direction.
    `mask_width` : `tuple[float | None]`
        The width of the mask in the x, y, and z directions.
        If `None`, the mask is not applied in that direction.
    `kx` : `int`
        The wavenumber in the x-direction.
    `ky` : `int`
        The wavenumber in the y-direction.
    `kz` : `int`
        The wavenumber in the z-direction.
    `s` : `int`
        The mode (0, 1, -1)
        0 => geostrophic mode
        1 => positive inertia-gravity mode
        -1 => negative inertia-gravity mode
    `phase` : `real`
        The phase of the wave. (Default: 0)
    
    Attributes
    ----------
    `omega` : `complex`
        The frequency of the wave (only for inertia-gravity modes).
    `period` : `float`
        The period of the wave (only for inertia-gravity modes).
    
    """
    def __init__(self, 
                 mset: nh.ModelSettings, 
                 mask_pos: tuple[float | None],
                 mask_width: tuple[float | None],
                 k: tuple[int],
                 s: int = 1, 
                 phase: float = 0, 
                 ) -> None:
        super().__init__(mset)

        # Shortcuts
        ncp = nh.config.ncp
        grid = mset.grid

        # Construct single wave
        z = nh.initial_conditions.SingleWave(mset, k, s, phase)

        if s != 0:
            self.omega = z.omega
            self.period = z.period

        # Construct mask
        mask = ncp.ones_like(grid.X[0])
        for x, pos, width in zip(grid.X, mask_pos, mask_width):
            if pos is not None and width is not None:
                mask *= ncp.exp(-(x - pos)**2 / width**2)

        # Apply mask
        z *= mask

        # Project onto the mode again
        q = grid.vec_q(s, use_discrete=True)
        p = grid.vec_p(s, use_discrete=True)
        z = z.project(p, q)

        # Inertia-gravity modes have to be multiplied by 2
        if s != 0:
            z *= 2

        # save the state
        self.fields = z.fields
        return
        