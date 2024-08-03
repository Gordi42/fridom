import fridom.nonhydro as nh


class Jet(nh.State):
    """
    A 3D jet with horizontal and vertical shear.
    
    Description
    -----------
    Superposition of a zonal jet and a geostrophic perturbation.
    Following the setup of Chouksey et al. 2022.
    For very large jet_strengths, convective instabilities can occur.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `jet_strength` : `float`
        The strength of the zonal jets.
    `jet_width` : `float`
        The width of the zonal jets.
    `pert_strength` : `float`
        The strength of the perturbation.
    `pert_wavenum` : `int`
        The wavenumber of the perturbation.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.

    Examples
    --------
    .. code-block:: python

        import fridom.nonhydro as nh
        # Set up the model settings
        fac = 7
        grid = nh.grid.cartesian.Grid(
            N=(2**fac, 2**fac, 2**(fac-3)), 
            L=(4, 4, 1), 
            periodic_bounds=(True, True, True))
        mset = nh.ModelSettings(grid=grid, f0=1, N2=1.0, dsqr=0.2**2, Ro=0.1)
        mset.time_stepper.dt = 2**(-fac) * 2
        mset.setup()
        # Create the initial conditions
        model = nh.Model(mset)
        model.z = nh.initial_conditions.Jet(
            mset, jet_strength=2, jet_width=0.16, 
            pert_strength=0.1, pert_wavenum=2)
        model.run(runlen=50.0)
    """
    def __init__(self, mset: nh.ModelSettings, 
                 jet_strength=1, 
                 jet_width=0.16,
                 pert_strength=0.05, 
                 pert_wavenum=5,
                 geo_proj=True):
        super().__init__(mset)
        ncp = nh.config.ncp

        X, Y, Z = mset.grid.X
        Lx, Ly, Lz = mset.grid.L

        # two opposite jets
        self.u.arr = -ncp.exp(-(Y-Ly/4)**2/(jet_width)**2)
        self.u.arr += ncp.exp(-(Y-3*Ly/4)**2/(jet_width)**2)
        self.u.arr *= jet_strength * ncp.cos(2*ncp.pi*Z/Lz)

        # add a small perturbation
        z_per = nh.initial_conditions.SingleWave(
            mset, kx=pert_wavenum, ky=0, kz=0, s=0)
        vel = (z_per.u**2 + z_per.v**2 + z_per.w**2)**0.5
        z_per /= vel.max()

        self.u.arr += pert_strength * z_per.u
        self.v.arr += pert_strength * z_per.v
        self.w.arr += pert_strength * z_per.w
        self.b.arr += pert_strength * z_per.b

        if geo_proj:
            proj_geo = nh.projection.GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.fields = z_geo.fields
        return