from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class Random(State):
    """
    Oceanic spectra with random phases.
    Used in the OBTA paper.
    """
    def __init__(self, grid:Grid, 
                 d=7, k0=6, seed=12345, amplitude_geostrophy=0.2, 
                 amplitude_wave=0.1, wave_power_law=-2) -> None:
        super().__init__(grid)

        # create the geostrophic field
        from fridom.shallowwater.initial_conditions \
            .geostrophic_spectra import GeostrophicSpectra
        z_geo = GeostrophicSpectra(grid, d, k0, seed=seed)

        # create the wave field
        from fridom.shallowwater.initial_conditions \
            .wave_spectra import WaveSpectra
        z_wav = WaveSpectra(grid, wave_power_law, seed=seed)

        # combine the two fields
        z = z_geo * amplitude_geostrophy + z_wav * amplitude_wave
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return


# remove symbols from namespace
del Grid, State
