from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class GeostrophicSpectra(State):
    """
    Oceanic spectra with random phases.
    """
    def __init__(self, grid:Grid, 
                 d=7, k0=6, seed=12345, random_type="normal") -> None:
        super().__init__(grid)
        # set coefficients for power law
        cp = self.cp
        mset = grid.mset
        
        # Define spectral function for geostrophic mode
        b = (7.+d)/4.
        a = (4./7.)*b-1
        def spectral_function(K):
            return K**7/(K**2 + a*k0**2)**(2*b)

        # Construct random phase field
        from fridom.shallowwater.initial_conditions \
            .random_phase import RandomPhase
        z = RandomPhase(grid, spectral_function, random_type, 1.0, seed)

        # Project onto geostrophic mode
        from fridom.shallowwater.projection import GeostrophicSpectral
        geo_proj = GeostrophicSpectral(grid)
        z = geo_proj(z)

        # Normalize
        max_amp = cp.amax(cp.abs(z.h))
        z /= max_amp

        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return


# remove symbols from namespace
del Grid, State