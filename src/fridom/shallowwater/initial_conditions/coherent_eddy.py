import fridom.shallowwater as sw

class CoherentEddy(sw.State):
    r"""
    Coherent barotropic eddy with Gaussian shape.

    Description
    -----------
    There are two versions of the coherent eddy. In the first version, the
    streamfunction of the eddy is given by an gaussian function:

    .. math::
        \psi = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}\right)

    where :math:`A` is the amplitude, :math:`(p_x, p_y)` is the relative 
    position of the eddy, :math:`(\sigma L_x)` is the width of the eddy, and
    :math:`L_x, L_y` are the domain sizes in the x and y directions. The
    velocity field is given by:

    .. math::
        u = \partial_y \psi, \quad v = -\partial_x \psi

    The second version of the coherent eddy prescribes the horizontal velocity
    as a gaussian function:

    .. math::
        \zeta = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}\right)

    Then the streamfunction is computed in spectral space:

    .. math::
        \hat{\psi} = \frac{\hat{\zeta}}{k_x^2 + k_y^2}

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `pos_x` : `float`, optional (default=0.5)
        The relative position of the eddy in the x-direction.
    `pos_y` : `float`, optional (default=0.5)
        The relative position of the eddy in the y-direction.
    `width` : `float`, optional (default=0.1)
        The relative width of the eddy. (relative to the domain size in the
        x-direction)
    `amplitude` : `float`, optional (default=1)
        The amplitude of the eddy. When the amplitude negative, the eddy
        rotates clockwise. Otherwise, it rotates counterclockwise.
    `gauss_field` : `str`, optional (default='vorticity')
        The field that is prescribed as a gaussian function. It can be either
        'vorticity' or 'streamfunction'.
    """
    def __init__(self, 
                 mset: sw.ModelSettings,
                 pos_x: float = 0.5,
                 pos_y: float = 0.5,
                 width: float = 0.1,
                 amplitude: float = 1,
                 gauss_field: str = 'vorticity'
                 ) -> None:
        super().__init__(mset)

        ncp = sw.config.ncp
        grid = self.grid
        Lx, Ly = grid.L

        FACE = sw.grid.AxisPosition.FACE
        position = sw.grid.Position((FACE, FACE))

        DIRICHLET = sw.grid.BCType.DIRICHLET
        bc_types = (DIRICHLET, DIRICHLET)

        field = sw.FieldVariable(
            mset, position=position, name="psi", bc_types=bc_types)

        X, Y = field.get_mesh()
        field.arr = amplitude * ncp.exp(
            -((X - pos_x * Lx)**2 + (Y - pos_y * Ly)**2) / (width*Lx)**2)

        if gauss_field == 'vorticity':
            kx, ky = grid.K
            k2 = kx**2 + ky**2
            psi = field.fft() / k2
            psi.arr = ncp.where(k2 == 0, 0, psi.arr)
            psi = psi.fft()
            self.psi = psi
        elif gauss_field == 'streamfunction':
            psi = field
        else:
            raise ValueError(f"Unknown gauss_field: {gauss_field}")

        self.p.arr = psi.arr * mset.f_coriolis.arr
        self.u.arr = - psi.diff(axis=1).arr
        self.v.arr = psi.diff(axis=0).arr
        
        return