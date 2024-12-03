from fridom.shallowwater.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class PolarizedWaveMaker(Module):
    """
    A polarized wave maker
    """
    def __init__(self, 
                 kx: float = 6.0,
                 ky: float = 0.0,
                 s: int = 1,
                 amplitude: float = 1.0,
                 mask_pos: tuple = (0.5, 0.5),
                 mask_width: tuple = (0.2, 0.2)):
        """
        Constructor of the wave maker source term.

        ## Arguments:
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            amplitude (float)     : The amplitude of the wave.
            mask_pos (tuple)      : The position of the mask.
            mask_width (tuple)    : The width of the mask.
        """
        super().__init__(name="Polarized Wave Maker",
                         kx=kx,
                         ky=ky,
                         s=s,
                         amplitude=amplitude,
                         mask_pos=mask_pos,
                         mask_width=mask_width)
    
    @start_module
    def start(self):
        # Shortcuts
        cp = self.grid.cp


        # Construct the polarized wave
        from fridom.shallowwater.initial_conditions import SingleWave
        z = SingleWave(self.grid, self.kx, self.ky, self.s)
        self.omega = z.omega.real


        # set mask
        mask = cp.ones_like(self.grid.X[0])
        for x, pos, width in zip(self.grid.X, self.mask_pos, self.mask_width):
            if pos is not None and width is not None:
                mask *= cp.exp(-(x - pos)**2 / width**2)

        mask *= self.amplitude

        z.u *= mask
        z.v *= mask
        z.h *= mask

        self.z_mask = z.copy()

        # project again on wave mode
        from fridom.shallowwater.eigenvectors import VecQ, VecP
        q = VecQ(self.s, self.grid)
        p = VecP(self.s, self.grid)
        z = z.fft()
        proj = z.dot(p)
        z_real = (q*proj).fft()
        z_imag = (q*(proj*(-1j))).fft()

        # save the state
        self.z_real = z_real
        self.z_imag = z_imag
        return
    
    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Update the state with the source term.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        cp = self.grid.cp
        phase = mz.clock.time * self.omega
        z = self.z_real * cp.cos(phase) + self.z_imag * cp.sin(phase)
        dz.u[:] += z.u
        dz.v[:] += z.v
        dz.h[:] += z.h
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    kx: {self.kx}\n"
        res += f"    ky: {self.ky}\n"
        res += f"    s: {self.s}\n"
        res += f"    amplitude: {self.amplitude}\n"
        res += f"    mask_pos: {self.mask_pos}\n"
        res += f"    mask_width: {self.mask_width}\n"
        return res
    
# remove symbols from namespace
del State, ModelState, Module, update_module, start_module