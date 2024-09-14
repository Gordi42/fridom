import fridom.nonhydro as nh

class CoherentEddy(nh.State):
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

    Examples
    --------
    This setup creates a coherent eddy in an scaled setup with varying 
    coriolis parameter. The eddy moves in positive x and 
    negative y direction and hits the northern wall.

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            N=(128, 128, 1), L=(3, 3, 1), periodic_bounds=(True, False, False))
        mset = nh.ModelSettings(grid=grid, f0=1, beta=0.2)
        mset.time_stepper.dt = 0.004
        mset.setup()
        model = nh.Model(mset)
        model.z = nh.initial_conditions.CoherentEddy(mset, width=0.15)
        model.run(runlen=np.timedelta64(20, 's'))

    The eddy becomes a rossby wave when either the amplitude is very small or
    the advection term is disabled:

    .. code-block:: python

        mset.tendencies.advection.disable()
    """
    def __init__(self, 
                 mset: nh.ModelSettings,
                 pos_x: float = 0.5,
                 pos_y: float = 0.5,
                 width: float = 0.1,
                 amplitude: float = 1,
                 gauss_field: str = 'streamfunction'
                 ) -> None:
        super().__init__(mset)

        ncp = nh.config.ncp
        grid = self.grid
        Lx, Ly, Lz = grid.L

        CENTER = nh.grid.AxisPosition.CENTER; FACE = nh.grid.AxisPosition.FACE
        position = nh.grid.Position((FACE, FACE, CENTER))

        DIRICHLET = nh.grid.BCType.DIRICHLET; NEUMANN = nh.grid.BCType.NEUMANN
        bc_types = (DIRICHLET, DIRICHLET, NEUMANN)

        field = nh.FieldVariable(
            mset, position=position, name="psi", bc_types=bc_types)

        X, Y, Z = field.get_mesh()
        field.arr = amplitude * ncp.exp(
            -((X - pos_x * Lx)**2 + (Y - pos_y * Ly)**2) / (width*Lx)**2)

        if gauss_field == 'vorticity':
            kx, ky, kz = grid.K
            k2 = kx**2 + ky**2
            psi = field.fft() / k2
            psi.arr = ncp.where(k2 == 0, 0, psi.arr)
            psi = psi.fft()
            self.psi = psi
        elif gauss_field == 'streamfunction':
            psi = field
        else:
            raise ValueError(f"Unknown gauss_field: {gauss_field}")

        self.u.arr = psi.diff(axis=1).arr
        self.v.arr = - psi.diff(axis=0).arr
        
        return