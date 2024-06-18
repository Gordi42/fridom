import numpy as np

from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class RandomPhase(State):
    """
    Calculates a random phase field with a presribed spectral scaling for
    the layer thickness h.
    """
    def __init__(self, grid:Grid, 
                 spectral_function, random_type="normal",
                 amplitude=1.0, seed=12345) -> None:
        """
        Arguments:
            spectral_function (callable) : with interface spectral_function(K)
            random_type (str)            : "uniform" or "normal"
                => uniform: The phase is randomized by multiplying the complex 
                            value e^ip to the state where p is uniformly
                            distributed between 0 and 2pi.
                => normal:  The phase is randomized by multiplying 
                            (a + ib) to the state where a and b are
                            normally distributed.
            amplitude (float)             : The resulting height field is
                                            normalized to this value.
            seed (int)                    : The seed for the random phase
        """
        super().__init__(grid)
        # get the wavenumber
        cp = self.cp
        mset = grid.mset
        Kx, Ky = tuple(grid.K)
        K = cp.sqrt(Kx**2 + Ky**2)
        k_hor = cp.sqrt(Kx**2 + Ky**2)

        # Define Function for random phase
        kx_flat = Kx.flatten(); ky_flat = Ky.flatten()
        k_order = cp.max(cp.abs(cp.array([kx_flat, ky_flat])), axis=0)
        angle = cp.angle(kx_flat + 1j*ky_flat)
        if mset.gpu:
            sort = np.lexsort((angle.get(), k_order.get()))
            sort = cp.array(sort)
        else:
            sort = cp.lexsort((angle, k_order))

        default_rng = cp.random.default_rng

        if random_type == "uniform":
            def random_phase(seed):
                # random phase between 0 and 2pi
                phase = default_rng(seed).uniform(0, 2*cp.pi, kx_flat.shape)
                return cp.exp(1j*phase).reshape(K.shape)

        elif random_type == "normal":
            def random_phase(seed):
                r = kx_flat*0 + 0j
                r[sort] = default_rng(seed).standard_normal(kx_flat.shape) + 1j*default_rng(2*seed).standard_normal(kx_flat.shape)
                return r.reshape(K.shape)
        else:
            raise ValueError("Unknown random phase type")

        kx_max = 2./3.*cp.amax(cp.abs(k_hor))
        large_k = (k_hor >= kx_max*1.0)
        spectra = cp.where(large_k, 0, spectral_function(K))
        # divide by K 
        spectra[K!=0] /= K[K!=0]

        from fridom.shallowwater.state import State
        z = State(grid, is_spectral=True)
        z.h[:] = cp.sqrt(spectra) * random_phase(seed)
        z = z.fft()

        # normalize
        z.h[:] -= cp.mean(z.h)
        scal = amplitude/cp.amax(z.h)
        z *= scal

        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return


# remove symbols from namespace
del Grid, State
