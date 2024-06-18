from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class WaveSpectra(State):
    """
    Wave spectra with power law scaling of frequency.
    """
    def __init__(self, grid:Grid, 
                 power_law=-2, seed=12345,
                 random_type="normal") -> None:
        super().__init__(grid)
        # get the wavenumber
        cp = self.cp
        mset = grid.mset

        def spectral_function(K):
            spectra = cp.sqrt(mset.f0 ** 2 + mset.csqr * K ** 2)
            spectra[spectra!=0] **= power_law
            return spectra

        # Construct random phase field
        from fridom.shallowwater.initial_conditions \
            .random_phase import RandomPhase
        z = RandomPhase(grid, spectral_function, random_type, 1.0, seed)

        # Remove geostrophic component
        from fridom.shallowwater.projection import GeostrophicSpectral
        geo_proj = GeostrophicSpectral(grid)
        z = z - geo_proj(z)

        # Normalize
        max_amp = cp.amax(cp.abs(z.h))
        z /= max_amp
        
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return


# remove symbols from namespace
del Grid, State
