import fridom.nonhydro as nh

class KelvinWave(nh.State):
    """
    Initial condition with a kelvin wave at the boundary.

    Description
    -----------
    Lets consider a Kelvin wave at the southern boundary of the domain with
    the horizontal wavenumber `kh` and the vertical wavenumber `kz`. Searching 
    in the linearized nonhydrostatic equations for a solution of the form:

    .. math::
        (U, V, W, B) \\exp(- k_n y) \\exp(i(k_h x + k_z z - \\omega t))
    
    yields

    .. math::
        \\omega = \\sqrt{\\frac{k_h^2 N^2}{k_h^2 + k_z^2}} \\quad
        k_n = \\frac{k_h f_0}{\\omega}

    .. math::
        U = - k_z \\quad
        V = 0 \\quad
        W = k_h \\quad
        B = -i N^2 k_h / \\omega
    
    Similar polarizations can be found for the other boundaries.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings object.
    `side` : `str`
        The side of the domain where the wave is located.
        Possible values are 'N': North, 'S': South, 'E': East, 'W': West.
    `kz` : `int`
        The vertical wavenumber. (Can be positive or negative)
    `kh` : `int`
        The horizontal wavenumber in the direction of the boundary. 
        (Must be positive)
    
    Examples
    --------
    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            N=[127]*3, L=[1]*3, periodic_bounds=(True, False, False))
        mset = nh.ModelSettings(grid=grid)
        mset.time_stepper.dt = np.timedelta64(10, 'ms')
        mset.setup()
        z = nh.initial_conditions.kelvin_wave(mset, 'N', kh=1, kz=2)
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=np.timedelta64(5, 's'))


    A vertical mode may be constructed with

    .. code-block:: python

        z  = nh.initial_conditions.kelvin_wave(mset, 'N', kh=1, kz=2)
        z += nh.initial_conditions.kelvin_wave(mset, 'N', kh=1, kz=-2)
    """
    def __init__(self, 
                 mset: nh.ModelSettings, 
                 side: str, 
                 kz: int, 
                 kh: int, 
                 phase: float = 0):
        super().__init__(mset, is_spectral=False)
    
        ncp = nh.config.ncp
        X, Y, Z = mset.grid.X
        Lx, Ly, Lz = mset.grid.L
        match side:
            case 'N':
                Lh = Lx; sign = 1;  Xn = Ly - Y; Xw = X
            case 'S':
                Lh = Lx; sign = -1; Xn = Y;      Xw = X
            case 'E':
                Lh = Ly; sign = -1; Xn = Lx - X; Xw = Y
            case 'W':
                Lh = Ly; sign = 1;  Xn = X;      Xw = Y

        kz = 2 * ncp.pi * kz / Lz
        kh = 2 * ncp.pi * kh / Lh
        om = ncp.sqrt((kh**2 * mset.N2) / (kh**2 + kz**2))

        wave = ncp.exp(1j * (kh * Xw + kz * Z + phase)) * ncp.sign(kz)
        wave *= ncp.exp(- mset.f0 * kh / om * Xn)

        # polarizations
        up = - kz
        wp = kh
        bp = 1j * sign * mset.N2 * kh / om

        # set the fields
        if side == "N" or side == "S":
            self.u.arr = (up * wave).real
        if side == "E" or side == "W":
            self.v.arr = (up * wave).real
        self.w.arr = (wp * wave).real
        self.b.arr = (bp * wave).real

        # save some parameters
        self.kh = kh
        self.kz = kz
        self.om = om
        return