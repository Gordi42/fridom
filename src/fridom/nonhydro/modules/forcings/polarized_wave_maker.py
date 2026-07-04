"""A polarized wave maker module for the non-hydrostatic model."""
from __future__ import annotations

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
        S(\boldsymbol{x}, t) =
            A \sin(\omega t) \boldsymbol{z}_W(\boldsymbol{x})

    where :math:`A` is the amplitude, :math:`\omega` is the frequency of
    the wave,
    that is computed from the dispersion relation (including
    discretization errors due to spatial and temporal discretization), and
    :math:`\boldsymbol{z}_W` is the WavePackage initial condition. The source
    term is added to the state vector tendencies:

    .. math::
        \partial_t \boldsymbol{z} \leftarrow \partial_t \boldsymbol{z}
                                              + S(\boldsymbol{x}, t)
    """

    #TODO(Silvano): Add documentation for the parameters
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

    def _on_setup(self) -> None:
        source = nh.initial_conditions.WavePackage(
            self.mset,
            mask_pos=self.position,
            mask_width=self.width,
            k=self.k)
        self.source = source * self.amplitude
        self.frequency = source.omega.real

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:  # noqa: D102
        ncp = fr.config.ncp
        mz.dz += self.source * ncp.sin(self.frequency * mz.clock.time)
        return mz

    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        res["position"] = self.position
        res["width"] = self.width
        res["frequency"] = self.frequency
        res["amplitude"] = self.amplitude
        return res
