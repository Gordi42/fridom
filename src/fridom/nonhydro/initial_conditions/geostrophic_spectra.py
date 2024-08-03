import fridom.nonhydro as nh

def geostrophic_energy_spectrum(kx, ky, kz, d=7, k0=6, c=2):
    """
    Geostrophic energy spectrum.
    
    Description
    -----------
    The geostrophic energy spectrum is separated into horizontal and vertical
    components. Following the work of Masur & Oliver [2020], the horizontal 
    energy spectrum :math:`S_h` is given by:

    .. math::
        S_h = \\frac{k^7}{\\left(k^2 + a k_0^2\\right)^{2b}}

    where :math:`k = \sqrt{k_x^2 + k_y^2}` is the horizontal wavenumber, :math:`a`
    and :math:`b` are constants:

    .. math::
        a = \\frac{4}{7}b - 1, \quad b = \\frac{7+d}{4}

    where :math:`d` is the power law exponent for large horizontal wavenumbers
    (:math:`S_h(k) \sim k^{-d}` for :math:`k \\to \\infty`). The parameter
    :math:`k_0` is the wavenumber with the maximum energy.

    The vertical component :math:`S_v` is given by:

    .. math::
        S_v = \\exp(-c|k_z|)

    where :math:`k_z` is the vertical wavenumber and :math:`c` is a constant.
    The total energy spectrum is given by :math:`S = S_h S_v`.
    
    Parameters
    ----------
    `kx` : `float`
        The horizontal wavenumber in the x-direction.
    `ky` : `float`
        The horizontal wavenumber in the y-direction.
    `kz` : `float`
        The vertical wavenumber.
    `d` : `float`, optional (default=7)
        The power law exponent for large horizontal wavenumbers 
        (:math:`S_h(k) \sim k^{-d}` for :math:`k \\to \\infty`).
    `k0` : `float`, optional (default=6)
        The wavenumber with the maximum energy.
    `c` : `float`, optional (default=2)
        The decay rate of the vertical energy spectrum.
    """
    ncp = nh.config.ncp
    
    # horizontal spectra
    kh = ncp.sqrt(kx**2 + ky**2)
    b = (7.+d)/4.
    a = (4./7.)*b-1
    h_spetra = kh**7/(kh**2 + a*k0**2)**(2*b)

    # vertical spectra
    v_spectra = ncp.exp(-c*ncp.abs(kz))
    return h_spetra * v_spectra


class RandomGeostrophicSpectra(nh.State):
    """
    Random geostrophic state with a given spectral energy density.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings (need to be set up).
    `seed` : `int`
        Seed for the random number generator (for the phase)
    `spectral_energy_density` : `callable(kx, ky, kz)`
        Callable that returns the spectral energy density as a function of the
        wavenumbers `kx`, `ky`, and `kz`.
    
    Examples
    --------
    .. code-block:: python

        import fridom.nonhydro as nh
        # Set up the model settings
        grid = nh.grid.cartesian.Grid(
            N=(128, 128, 32), L=(10, 10, 1), 
            periodic_bounds=(False, True, False))
        mset = nh.ModelSettings(grid=grid, f0=1, N2=1.0, dsqr=0.2**2)
        mset.time_stepper.dt = 0.1
        mset.setup()
        # Create the initial conditions
        ic = nh.initial_conditions
        def spectra(kx, ky, kz):
            return ic.geostrophic_energy_spectrum(kx, ky, kz, c=0.2, k0=1.4, d=7)
        z = ic.RandomGeostrophicSpectra(mset, spectral_energy_density=spectra) * 0.1
        # Create and run the model
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=200.0)

    """
    def __init__(self, 
                 mset: nh.ModelSettings,
                 seed=12345,
                 spectral_energy_density=geostrophic_energy_spectrum):
        super().__init__(mset, is_spectral=False)

        ncp = nh.config.ncp
        kx, ky, kz = mset.grid.K
        shape = kx.shape

        # construct the geostrophic eigenvectors
        q = mset.grid.vec_q(s=0, use_discrete=True)

        # scale the geostrophic eigenvector such that they have energy 1
        abs = ncp.absolute
        dsqr = self.mset.dsqr
        N2 = self.mset.N2
        # calculate spectral energy using Parseval's theorem
        energy = 0.5 * (   abs(q.u.arr)**2 
                         + abs(q.v.arr)**2 
                         + abs(q.w.arr)**2 * dsqr
                         + abs(q.b.arr)**2 / N2 )

        energy = ncp.where(energy == 0, 1, energy)
        
        q /= ncp.sqrt(energy)

        # construct a random phase
        p1 = nh.utils.random_array(shape, seed)
        p2 = nh.utils.random_array(shape, 2*seed)
        r = p1 + 1j * p2

        # construct the spectral energy density
        spectra = spectral_energy_density(kx, ky, kz)

        # construct the geostrophic state
        z = q * r * ncp.sqrt(spectra)

        # transform to physical space and normalize the state such that the
        # maximum velocity is 1
        z = z.fft()
        scal = 1 / ncp.amax(z.u.arr)
        z *= scal

        # set the state
        self.fields = z.fields
        return