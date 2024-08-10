import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class PolarizedWaveMaker(fr.modules.Module):
    r"""
    A wave maker that creates a polarized wave.

    Description
    -----------
    The source term is constructed from WavePackage initial condition (see
    :py:class:`fridom.nonhydro.initial_conditions.WavePackage`):

    .. math::
        S(\boldsymbol{x}, t) = A \sin(\omega t) \boldsymbol{z}_W(\boldsymbol{x})

    where :math:`A` is the amplitude, :math:`\omega` is the frequency of the wave,
    that is computed from the dispersion relation (including 
    discretization errors due to spatial and temporal discretization), and
    :math:`\boldsymbol{z}_W` is the WavePackage initial condition. The source
    term is added to the state vector tendencies:

    .. math::
        \partial_t \boldsymbol{z} \leftarrow \partial_t \boldsymbol{z} 
                                              + S(\boldsymbol{x}, t)
    
    Examples
    --------

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            N=(512, 1, 512), L=(200, 1, 200), periodic_bounds=(True, True, True))
        mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
        mset.time_stepper.dt = np.timedelta64(1, 'm')
        mset.tendencies.advection.disable()

        # create a NetCDF writer to save the output
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.etot, mz.z.ekin],
            write_interval = np.timedelta64(20, 'm')))


        mset.tendencies.add_module(nh.modules.forcings.PolarizedWaveMaker(
            position = (50, None, 100),
            width = (10, None, 10),
            k = (40, 0, -40)))

        mset.tendencies.add_module(s)
        mset.setup()

        model = nh.Model(mset)
        model.run(runlen=np.timedelta64(2, 'D'))

    When changing the boundary condition in z to non-periodic. The wave in the
    wave maker would be a vertical mode. The wave ray would then propagate 
    upwards and downwards. To create a purely downward / upward propagating wave
    ray, we need to apply a trick by creating the wave maker for a periodic
    grid and then insert the wave maker into the non-periodic setup. 
    The following code demonstrates this for the above example:

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            N=(512, 1, 512), L=(200, 1, 200), periodic_bounds=(True, True, False))
        mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
        mset.time_stepper.dt = np.timedelta64(1, 'm')
        mset.tendencies.advection.disable()

        # create a NetCDF writer to save the output
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.etot, mz.z.ekin],
            write_interval = np.timedelta64(20, 'm')))
        mset.setup()

        # Add the source term (but from a new modelsettings with periodic boundaries)
        grid_periodic = nh.grid.cartesian.Grid(
            N=(512, 1, 512), L=(200, 1, 200), periodic_bounds=(True, True, True))
        mset_periodic = nh.ModelSettings(grid=grid_periodic, f0=1e-4, N2=2.5e-5)

        s = nh.modules.forcings.PolarizedWaveMaker(
            position = (50, None, 100),
            width = (10, None, 10),
            k = (40, 0, -40))
        mset_periodic.tendencies.add_module(s)
        mset_periodic.setup()

        # Now we add the source term to the original modelsettings
        mset.tendencies.add_module(s)

        model = nh.Model(mset)
        model.run(runlen=np.timedelta64(2, 'D'))
    """
    name = "Polarized Wave Maker"

    def __init__(self, 
                 position: tuple[float | None],
                 width: tuple[float | None],
                 k: tuple[int],
                 amplitude: float = 1.0,
                 ) -> None:
        super().__init__()
        self.position = position
        self.width = width
        self.amplitude = amplitude
        self.frequency = None
        self.k = k
        return

    @fr.modules.module_method
    def setup(self, mset: 'nh.ModelSettings'):
        super().setup(mset)
        source = nh.initial_conditions.WavePackage(
            mset, 
            mask_pos=self.position,
            mask_width=self.width,
            k=self.k)
        self.source = source * self.amplitude
        self.frequency = source.omega.real
        return

    @fr.utils.jaxjit
    def add_source_term(self, dz: nh.State, time: float) -> nh.State:
        ncp = fr.config.ncp
        dz += self.source * ncp.sin(self.frequency * time)
        return dz

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.add_source_term(mz.dz, mz.time)
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["position"] = self.position
        res["width"] = self.width
        res["frequency"] = self.frequency
        res["amplitude"] = self.amplitude
        return res
