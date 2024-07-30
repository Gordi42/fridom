import fridom.nonhydro as nh

class SingleWave(nh.State):
    """
    An initial condition that consist of a single wave with a
    given wavenumber and a given mode.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
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
    `phase` : `float`
        The phase of the wave. (default: 0)
    `use_discrete` : `bool` (default: True)
        Whether to use the discrete eigenvectors or the analytical ones.
    
    Attributes
    ----------
    `z` : `State`
        The state vector
    `omega` : `complex`
        The frequency of the wave (includes effects of time discretization)
        (only for inertia-gravity modes).
    `period` : `float`
        The period of the wave (includes effects of time discretization)
        (only for inertia-gravity modes).

    Examples
    --------
    >>> import fridom.nonhydro as nh
    >>> import numpy as np
    >>> grid = nh.grid.cartesian.Grid(
    ...     N=[127]*3, L=[1]*3, periodic_bounds=(True, True, True))
    >>> mset = nh.ModelSettings(grid=grid, dsqr=0.02)
    >>> mset.time_stepper.dt = np.timedelta64(10, 'ms')
    >>> mset.setup()
    >>> z = nh.initial_conditions.SingleWave(mset, kx=2, ky=0, kz=1)
    >>> model = nh.Model(mset)
    >>> model.z = z
    >>> model.run(runlen=np.timedelta64(10, 's'))
    """
    def __init__(self, 
                 mset: nh.ModelSettings, 
                 kx: int = 2, 
                 ky: int = 0, 
                 kz: int = 1, 
                 s: int = 1, 
                 phase: float = 0, 
                 use_discrete: bool = True) -> None:
        super().__init__(mset, is_spectral=False)

        # Shortcuts
        ncp = nh.config.ncp
        grid = mset.grid
        Kx, Ky, Kz = grid.K
        Lx, Ly, Lz = grid.L
        pi = ncp.pi

        # Find index of the wavenumber in the grid (nearest neighbor)
        kx = 2*pi*kx/Lx; ky = 2*pi*ky/Ly; kz = 2*pi*kz/Lz
        k_loc = (Kx == kx) & (Ky == ky) & (Kz == kz)

        # Construct the spectral field of the corresponding mode
        # all zeros except for the mode
        mask = ncp.where(k_loc, 1, 0)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            from fridom.nonhydro.eigenvectors import VecQ
            q = VecQ(s, mset)
        else:
            from fridom.nonhydro.eigenvectors import VecQAnalytical
            q = VecQAnalytical(s, mset)

        # Construct the state
        z = (q * mask * ncp.exp(1j*phase)).fft()

        # Normalize the state
        z /= z.norm_l2()
        
        # Set the state
        self.fields = z.fields

        # Calculate the frequency and period of the wave
        om = grid.omega((kx, ky, kz), use_discrete=True)
        ts = mset.time_stepper
        self.omega = ts.time_discretization_effect(om)
        self.period = 2*pi/self.omega.real

        # save the wavenumbers
        self.kx = kx
        self.ky = ky
        self.kz = kz
        return
        
