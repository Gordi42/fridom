"""A gaussian wave maker module."""
from __future__ import annotations

import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class GaussianWaveMaker(fr.modules.Module):

    r"""
    A Gaussian wave maker that forces the u-component of the velocity field.

    Description
    -----------
    Creates a gaussian source term of the form:

    .. math::
        M(\boldsymbol{x}) =
            \prod_{i=1}^{3} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)

    .. math::
        S(\boldsymbol{x}, t) = A \sin(2\pi f t) M(\boldsymbol{x})

    where :math:`A` is the amplitude, :math:`x_i` is the x coordinate,
    :math:`p_i` is the position, :math:`w_i` is the width and :math:`f`
    is the frequency of the wave maker. The source term is added to the
    u-component of the velocity field:

    .. math::
        \partial_t u \leftarrow \partial_t u + S(\boldsymbol{x}, t)

    Parameters
    ----------
    position : tuple[float | None]
        The position of the wave maker (center of the gaussian).
        The wave maker is constant over axis with `position[axis]=None`.
    width : tuple[float | None]
        The width of the wave maker (width of the gaussian).
        The wave maker is constant over axis with `width[axis]=None`.
    frequency : float
        The frequency of the wave maker.
    amplitude : float
        The amplitude of the wave maker.
    variable : str
        The variable to force. (Default: "u")

    """

    name = "Gaussian Wave Maker"

    def __init__(self,
                 position: tuple[float | None],
                 width: tuple[float | None],
                 frequency: float,
                 amplitude: float,
                 variable: str = "u") -> None:
        super().__init__()
        self.position = position
        self.width = width
        self.frequency = frequency
        self.amplitude = amplitude
        self.variable = variable

    def _on_setup(self) -> None:
        ncp = fr.config.ncp
        # Construct mask
        mask = ncp.ones_like(self.grid.x_mesh[0])
        for x, pos, width in zip(self.grid.x_mesh, self.position,
                                 self.width, strict=False):
            if pos is not None and width is not None:
                mask *= ncp.exp(-(x - pos)**2 / width**2)
        mask *= self.amplitude
        self.mask = mask

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:  # noqa: D102
        ncp = fr.config.ncp
        tendency = self.mask * ncp.sin(
            2 * ncp.pi * self.frequency * mz.clock.time)
        mz.dz.fields[self.variable] += tendency
        return mz

    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        res["position"] = self.position
        res["width"] = self.width
        res["frequency"] = self.frequency
        res["amplitude"] = self.amplitude
        return res
