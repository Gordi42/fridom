"""Spectral advection for the nonhydrostatic model."""
import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class SpectralAdvection(fr.modules.advection.AdvectionBase):

    """
    Advection Scheme for the spectral grid.

    Parameters
    ----------
    padding : fr.grid.FFTPadding
        Padding to use for the FFT operations.

    """

    name = "Spectral Advection"

    def __init__(self, padding: fr.grid.FFTPadding = fr.grid.FFTPadding.TRIM) -> None:
        super().__init__()
        self.padding = padding

    def _on_setup(self) -> None:
        if hasattr(self.mset, "rossby_number"):
            self.scaling = self.mset.rossby_number

    def advect_state(self, z: nh.State, dz: nh.State) -> nh.State:  # noqa: D102
        divergence = self.diff_module.div
        padding = self.padding
        zp = z.ifft(padding=padding)

        uu, uv, uw = (zp.u * zp.velocity).fft(padding=padding)
        vu, vv, vw = uv, *(zp.v * zp.velocity[1:]).fft(padding=padding)
        wu, wv, ww = uw, vw, (zp.w * zp.w).fft(padding=padding)

        dz.u -= self.scaling * divergence([uu, vu, wu])
        dz.v -= self.scaling * divergence([uv, vv, wv])
        dz.w -= self.scaling * divergence([uw, vw, ww])

        # now do all the tracer advection
        for field in zp.tracers:
            if field.flags["NO_ADV"]:
                continue

            flux = (field * zp.velocity).fft(padding=padding)
            dz[field.name] -= self.scaling * divergence(flux)
        return dz

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def padding(self) -> fr.grid.FFTPadding:
        """Padding to use for the FFT operations."""
        return self._padding

    @padding.setter
    def padding(self, padding: fr.grid.FFTPadding) -> None:
        self._padding = padding
