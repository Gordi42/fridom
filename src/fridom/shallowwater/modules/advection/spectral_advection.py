"""Spectral advection scheme for the shallow water equations."""
from __future__ import annotations

import fridom.framework as fr
import fridom.shallowwater as sw


@fr.utils.jaxify
class SpectralAdvection(fr.modules.advection.AdvectionBase):

    r"""
    Spectral advection scheme for the shallow water equations.

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
        if hasattr(self.mset, "Ro"):
            self.scaling = self.mset.Ro

    def advect_state(self, z: sw.State, dz: sw.State) -> sw.State:  # noqa: D102

        def fft(x: fr.FieldBase) -> fr.FieldBase:
            return x.fft(padding=self.padding)

        def ifft(x: fr.FieldBase) -> fr.FieldBase:
            return x.ifft(padding=self.padding)

        grad = self.diff_module.grad
        divergence = self.diff_module.div

        vel_p = ifft(z.velocity)
        dz.u -= self.scaling * fft(vel_p @ ifft(grad(z.u)))
        dz.v -= self.scaling * fft(vel_p @ ifft(grad(z.v)))
        dz.p -= self.scaling * divergence(fft(vel_p * ifft(z.p)))

        for field in z.tracers:
            if field.flags["NO_ADV"]:
                continue
            dz[field.name] -= self.scaling * fft(vel_p @ ifft(grad(field)))

        return dz
